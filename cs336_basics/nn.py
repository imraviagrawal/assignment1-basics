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

class Rope(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len


        if self.theta != 0:
            i_vec = torch.arange(max_seq_len, device=device)[:, None]
            k_vec = torch.arange(d_k//2, device=device)[None, :]

            thetas = i_vec / theta ** (2*k_vec/d_k)
            R = torch.stack((thetas.cos(), thetas.sin()))

            R = R.to(device=device)
            self.register_buffer("R", R, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # basic Version 
        if self.theta == 0:
            return x
        
        seq_len = x.shape[-2] # last dim, (batch, seq, token id)
        if token_positions is None:
            even = x[..., ::2]
            odd = x[..., 1::2]
            c = self.R[0, :seq_len ,...]
            s = self.R[1, :seq_len ,...]
            tmp = even*s + odd*c
            x[..., ::2] = even*c - odd*s
            x[..., 1::2] = tmp

        else:
            even = x[..., token_positions, ::2]
            odd = x[..., token_positions, 1::2]

            c = self.R[0, token_positions, ...]
            s = self.R[1, token_positions, ...]
            tmp = even*s + odd*c
            x[...,token_positions,::2] = even * c - odd *s
            x[..., token_positions, 1::2] = tmp
        return x
import torch
import torch.nn as nn

class Rope(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even"
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        if theta != 0:
            i = torch.arange(max_seq_len, device=device)[:, None]         # positions
            k = torch.arange(d_k // 2, device=device)[None, :]            # pair index
            angles = i / (theta ** (2 * k / d_k))
            cos = angles.cos()
            sin = angles.sin()
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        if self.theta == 0:
            return x

        if token_positions is None:
            c = self.cos[:x.size(-2)]   # (seq_len, d_k//2)
            s = self.sin[:x.size(-2)]
        else:
            c = self.cos.index_select(0, token_positions)
            s = self.sin.index_select(0, token_positions)

        even = x[..., ::2]
        odd  = x[..., 1::2]
        x_rotated = torch.empty_like(x)
        x_rotated[..., ::2] = even * c - odd * s
        x_rotated[..., 1::2] = even * s + odd * c
        return x_rotated

class Softmax(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim=dim


    def forward(self, x):
        orig_dtype = x.dtype

        x = x.float() # full precision
        # scale so that large value does not lead to inf value 
        max_val, _ = x.max(dim=self.dim, keepdim=True) # last dim max 
        
        # exp 
        x_scaled = torch.exp(x - max_val) # exp(x) 
        
        # sum of exp 
        x_sum = torch.sum(x_scaled, dim=self.dim, keepdim=True) # sum of exp(x) along last dim

        out = x_scaled / x_sum
        
        return out.to(orig_dtype) # exp(x) / sum(exp(x)), may be need broadcase
