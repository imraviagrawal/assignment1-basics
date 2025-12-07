import torch
import math
import numpy as np
import torch.nn as nn
import einops

# class LayerNorm(nn.Module):
#     def __init__(self, d_model, eps=1e-5):
#         super().__init__()
#         self.eps = eps
#         self.gamma = nn.Parameter(torch.ones(d_model))
#         self.beta = nn.Parameter(torch.zeros(d_model))

#     def forward(self, x: torch.Tensor):
#         # calculate mean 
#         d_model = x.shape[-1]
#         x_flattened = x.reshape(-1, d_model)
#         mean = x_flattened.mean(dim=-1, keepdim=True) # last dim mean 
#         var = (1/d_model)*torch.sum((x_flattened - mean)**2, dim=-1, keepdim=True) 
#         x_hat = (x - mean)*torch.rsqrt(var**2 + self.eps) # zero mean and scaled by variance 
#         return x_hat*self.gamma + self.beta
    
import math
import pytest
import torch
import torch.nn as nn

# ---------- User implementation (as provided) ----------
class UserLayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gemma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor):
        # calculate mean 
        d_model = x.shape[-1]
        mean = x.mean(dim=-1, keepdim=True) # last dim mean 
        var = (1/d_model)*torch.sum((x - mean)**2, dim=-1, keepdim=True) 
        x_hat = (x - mean)*torch.rsqrt(var + self.eps) # zero mean and scaled by variance 
        return x_hat*self.gemma + self.beta

# ---------- Helpers ----------
DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")
try:
    if torch.backends.mps.is_available():
        DEVICES.append("mps")
except Exception:
    pass

DTYPES = [torch.float32]
if torch.cuda.is_available() or getattr(torch.backends, "mps", None):
    # half precision tests only where supported
    DTYPES.append(torch.bfloat16)
    DTYPES.append(torch.float16)

SHAPES = [
    (8, 16),        # 2D [B, C]
    (4, 7, 32),     # 3D [B, T, C]
    (2, 3, 5, 64),  # 4D [N, B, T, C]
]

def make_pair(d_model, eps=1e-6, elementwise_affine=True):
    """Return (user_ln, torch_ln) with matched params and eps."""
    u = UserLayerNorm(d_model, eps=eps)
    t = nn.LayerNorm(d_model, eps=eps, elementwise_affine=elementwise_affine)
    if elementwise_affine:
        with torch.no_grad():
            # align params: ones/zeros (already default)
            t.weight.fill_(1.0)
            t.bias.zero_()
    return u, t

def _rand_like(shape, device, dtype):
    # Keep variance reasonable to avoid overflow in half types
    torch.manual_seed(0)
    x = torch.randn(shape, device=device, dtype=torch.float32)
    if dtype != torch.float32:
        x = x.to(dtype)
    return x

# ---------- Tests ----------

@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("eps", [1e-5, 1e-6, 1e-3])
def test_output_parity(shape, device, dtype, eps):
    C = shape[-1]
    user_ln, torch_ln = make_pair(C, eps=eps, elementwise_affine=True)
    user_ln = user_ln.to(device=device, dtype=dtype)
    torch_ln = torch_ln.to(device=device, dtype=dtype)

    x = _rand_like(shape, device, dtype)

    y_user = user_ln(x)
    y_torch = torch_ln(x)

    # Tolerances: looser for low-precision types
    atol = 1e-5 if dtype == torch.float32 else 2e-3
    rtol = 1e-4 if dtype == torch.float32 else 1e-3
    assert torch.allclose(y_user, y_torch, atol=atol, rtol=rtol), f"Mismatch for shape={shape}, device={device}, dtype={dtype}, eps={eps}"

@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("device", DEVICES)
def test_grad_parity(shape, device):
    C = shape[-1]
    dtype = torch.float32
    user_ln, torch_ln = make_pair(C, eps=1e-5, elementwise_affine=True)
    user_ln = user_ln.to(device=device, dtype=dtype)
    torch_ln = torch_ln.to(device=device, dtype=dtype)

    # Input that requires grad
    x = _rand_like(shape, device, dtype).requires_grad_(True)
    x2 = x.detach().clone().requires_grad_(True)

    # Simple scalar losses so we can compare grads
    out_user = user_ln(x).sum()
    out_torch = torch_ln(x2).sum()
    out_user.backward()
    out_torch.backward()

    atol = 1e-5
    rtol = 1e-4
    assert torch.allclose(x.grad, x2.grad, atol=atol, rtol=rtol), "Input gradient mismatch"
    assert torch.allclose(user_ln.gemma.grad, torch_ln.weight.grad, atol=atol, rtol=rtol), "Weight gradient mismatch"
    assert torch.allclose(user_ln.beta.grad, torch_ln.bias.grad, atol=atol, rtol=rtol), "Bias gradient mismatch"

@pytest.mark.parametrize("device", DEVICES)
def test_affine_disabled_output(device):
    """When PyTorch disables affine, match by fixing user params to 1/0."""
    C = 32
    dtype = torch.float32
    eps = 1e-5
    user_ln, torch_ln = make_pair(C, eps=eps, elementwise_affine=False)
    user_ln = user_ln.to(device=device, dtype=dtype)
    torch_ln = torch_ln.to(device=device, dtype=dtype)

    # Freeze user params to simulate 'no affine'
    user_ln.gemma.requires_grad_(False)
    user_ln.beta.requires_grad_(False)
    with torch.no_grad():
        user_ln.gemma.fill_(1.0)
        user_ln.beta.zero_()

    x = _rand_like((4, 7, C), device, dtype)
    y_user = user_ln(x)
    y_torch = torch_ln(x)

    assert torch.allclose(y_user, y_torch, atol=1e-6, rtol=1e-5)

@pytest.mark.parametrize("device", DEVICES)
def test_mean_zero_unit_var_last_dim(device):
    """Check that the user LN outputs ~zero mean and unit variance across the last dim."""
    C = 48
    x = _rand_like((5, 6, C), device, torch.float32)
    user_ln = UserLayerNorm(C, eps=1e-5).to(device)
    y = user_ln(x)
    # stats along last dim
    mean = y.mean(dim=-1)
    var = y.var(dim=-1, unbiased=False)
    assert mean.abs().max() < 2e-6
    assert (var - 1.0).abs().max() < 2e-5

@pytest.mark.parametrize("device", DEVICES)
def test_multidim_normalized_shape_difference(device):
    """Show a scenario PyTorch supports but the user impl does not: normalize over (T, C)."""
    shape = (2, 3, 4)  # (B, T, C)
    x = _rand_like(shape, device, torch.float32)
    # PyTorch over last two dims
    ln_tc = nn.LayerNorm((shape[-2], shape[-1]), eps=1e-5).to(device)
    # User impl only over last dim
    ln_user = UserLayerNorm(shape[-1], eps=1e-5).to(device)

    y_tc = ln_tc(x)
    y_user = ln_user(x)

    # They should not match in general
    with pytest.raises(AssertionError):
        assert torch.allclose(y_tc, y_user, atol=1e-6, rtol=1e-5), "Unexpected match; should differ when normalizing over (T, C) vs only C."

@pytest.mark.parametrize("device", DEVICES)
def test_numerical_stability_on_near_constant_inputs(device):
    """Stress test with tiny variance; verify finite outputs and parity with PyTorch."""
    C = 32
    base = torch.full((8, C), 1e3, device=device, dtype=torch.float32)
    noise = torch.full_like(base, 1e-7)  # extremely tiny variance
    x = base + noise * torch.randn_like(base)

    eps = 1e-5
    u, t = make_pair(C, eps=eps)
    u = u.to(device)
    t = t.to(device)

    yu = u(x)
    yt = t(x)
    assert torch.isfinite(yu).all()
    assert torch.isfinite(yt).all()
    assert torch.allclose(yu, yt, atol=5e-6, rtol=1e-5)

@pytest.mark.parametrize("device", DEVICES)
def test_large_magnitude_inputs(device):
    """Stress test large magnitudes."""
    C = 64
    x = (1e4 * torch.randn(16, C, device=device, dtype=torch.float32))
    u, t = make_pair(C, eps=1e-5)
    u = u.to(device)
    t = t.to(device)

    yu = u(x)
    yt = t(x)
    assert torch.isfinite(yu).all() and torch.isfinite(yt).all()
    assert torch.allclose(yu, yt, atol=3e-5, rtol=1e-5)

def test_state_dict_compatibility():
    """The user impl uses non-standard param names, which breaks direct state_dict compatibility."""
    C = 8
    u = UserLayerNorm(C)
    t = nn.LayerNorm(C)
    sd_u = u.state_dict()
    sd_t = t.state_dict()
    # Key sets should differ (gamma/beta vs weight/bias)
    assert set(sd_u.keys()) != set(sd_t.keys()), "Param names unexpectedly match; this test documents the difference."

if __name__ == "__main__":
    # Allow running as a script for a quick smoke test
    failed = pytest.main(["-q"])
    raise SystemExit(failed)
