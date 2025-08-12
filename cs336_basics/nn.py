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
