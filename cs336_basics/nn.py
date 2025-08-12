import torch
import math
import numpy as np
import torch.nn as nn
import einops


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        std = math.sqrt(2 / (in_features + out_features))
        weight = torch.empty(out_features, in_features, device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3, b=3)
        self.weight = nn.Parameter(weight) # d_in, d_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: batch, seq, d_modal
        output = einops.einsum(x, self.weight, "... d_in, out_features d_in -> ... out_features")
        return output
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        weight = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        self.param = nn.Parameter(torch.nn.init.trunc_normal_(weight, a=-3, b=3))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.param[token_ids]
    
class RMSnorm(nn.Module):
    def __init__(self, d_modal:int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_modal = d_modal
        self.eps = eps
        self.param = nn.Parameter(torch.ones(d_modal, device=device, dtype=dtype)) # d_modal
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # upscale input to float32 to avoid overflow
        in_type = x.dtype
        x = x.to(torch.float32)

        norm = x.square().mean(dim=-1, keepdim=True)
        result = (x*self.param)/torch.sqrt(norm - self.eps)
        # result = (x*torch.sqrt(1/))
        return result.to(in_type)

def silu(param: torch.Tensor): 
    return param*torch.sigmoid(param) # implement

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff = None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        if d_ff == None:
            d_ff = int(np.ceil(8*d_model // 3 / 64) * 64)
        self.w1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.w2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)
        self.w3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        w_1x = self.w1.forward(x) # d_model, d_ff
        w_3x = self.w3.forward(x) # d_model, d_ff
        return self.w2.forward(silu(w_1x)*w_3x)
