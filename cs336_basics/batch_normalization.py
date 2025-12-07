# my_batchnorm_llm.py
import torch
from torch import nn

class MyBatchNormLLM(nn.Module):
    """
    Scratch BatchNorm for inputs shaped (B, T, C):
      - B: batch size
      - T: sequence length
      - C: hidden size (#features)

    Behavior should match torch.nn.BatchNorm1d applied to (B*T, C).

    Args:
      num_features: C
      eps: float (default 1e-5)
      momentum: float or None
      affine: bool (learn weight/bias)
      track_running_stats: bool (store running_mean/var)
    """
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()
        self.mean = torch.zeros(num_features, requires_grad=False)
        self.variance = torch.zeros(num_features, requires_grad=False)
        self.gemma = torch.ones(num_features)
        self.lambda = torch.zeros(num_features)
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C)
        Normalize along C using stats computed across B*T.
        Return: same shape (B, T, C).
        """
        c = x.shape[-1]
        batch_mean = torch.mean(x.reshape(-1, c), dim=0)
        # batch_std = torch.std(x.reshape(-1, c), dim=0)
        batch_std = ((x - batch_mean)**2).mean(dim=0)
        if self.track_running_stats:
            self.mean = self.momentum*self.mean + (1- self.momentum)*batch_mean
            self.std = self.momentum*self.std + (1- self.momentum)*batch_std
        x_hat = (x - batch_mean)/torch.sqrt(batch_std + self.eps)
        return self.gemma*x_hat + self.lambda

    
# test_my_batchnorm_llm.py
import pytest
import torch
from torch import nn
# from my_batchnorm_llm import MyBatchNormLLM

def _devices():
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    return devs

def _rand_input(shape, device="cpu", dtype=torch.float32, seed=0):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return torch.randn(*shape, generator=g, device=device, dtype=dtype) * 3

# -----------------------
# Core tests
# -----------------------

@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("B,T,C", [(2,4,8), (3,5,16), (1,7,32)])
@pytest.mark.parametrize("eps", [1e-5, 1e-3])
@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("track_running_stats", [True, False])
def test_forward_matches_batchnorm1d(device, B, T, C, eps, affine, track_running_stats):
    x = _rand_input((B, T, C), device=device)

    my = MyBatchNormLLM(C, eps=eps, affine=affine, track_running_stats=track_running_stats).to(device)
    ref = nn.BatchNorm1d(C, eps=eps, affine=affine, track_running_stats=track_running_stats).to(device)

    # Sync params if affine
    if affine:
        with torch.no_grad():
            my.weight.copy_(ref.weight)
            my.bias.copy_(ref.bias)

    my.train()
    ref.train()

    # Reference needs input as (N, C, L) → we flatten batch/seq
    x_ref = x.permute(0,2,1).reshape(B, C, T)  # (B, C, T)
    y_ref = ref(x_ref).permute(0,2,1)  # back to (B, T, C)

    y_my = my(x)
    assert y_my.shape == x.shape
    assert torch.allclose(y_my, y_ref, atol=1e-5, rtol=1e-5), "Mismatch with BatchNorm1d"

@pytest.mark.parametrize("device", _devices())
def test_running_stats_update(device):
    B, T, C = 2, 3, 4
    x = _rand_input((B, T, C), device=device)

    my = MyBatchNormLLM(C, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True).to(device)
    ref = nn.BatchNorm1d(C, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True).to(device)

    my.train(); ref.train()

    for _ in range(3):
        xb = _rand_input((B, T, C), device=device)
        _ = my(xb)
        _ = ref(xb.permute(0,2,1))  # BN1d expects (N,C,L)

    assert torch.allclose(my.running_mean, ref.running_mean, atol=1e-5, rtol=1e-5)
    assert torch.allclose(my.running_var,  ref.running_var,  atol=1e-5, rtol=1e-5)

@pytest.mark.parametrize("device", _devices())
def test_backward_grads_match(device):
    B, T, C = 4, 5, 6
    x1 = _rand_input((B, T, C), device=device).requires_grad_(True)
    x2 = x1.clone().detach().requires_grad_(True)

    my = MyBatchNormLLM(C, eps=1e-5, affine=True, track_running_stats=True).to(device)
    ref = nn.BatchNorm1d(C, eps=1e-5, affine=True, track_running_stats=True).to(device)

    with torch.no_grad():
        my.weight.copy_(ref.weight)
        my.bias.copy_(ref.bias)

    my.train(); ref.train()

    y1 = my(x1)
    y2 = ref(x2.permute(0,2,1)).permute(0,2,1)

    loss1 = (y1**2).mean()
    loss2 = (y2**2).mean()

    loss1.backward()
    loss2.backward()

    assert torch.allclose(x1.grad, x2.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(my.weight.grad, ref.weight.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(my.bias.grad, ref.bias.grad, atol=1e-5, rtol=1e-5)

@pytest.mark.parametrize("device", _devices())
def test_eval_uses_running_stats(device):
    B, T, C = 2, 3, 4
    x = _rand_input((B, T, C), device=device)

    my = MyBatchNormLLM(C, track_running_stats=True).to(device)
    ref = nn.BatchNorm1d(C, track_running_stats=True).to(device)

    my.train(); ref.train()
    _ = my(x); _ = ref(x.permute(0,2,1))

    my.eval(); ref.eval()
    y_my = my(x)
    y_ref = ref(x.permute(0,2,1)).permute(0,2,1)

    assert torch.allclose(y_my, y_ref, atol=1e-5, rtol=1e-5)
