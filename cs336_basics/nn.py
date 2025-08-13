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
        result = (x*self.param)/torch.sqrt(norm + self.eps)
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

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        # self.token_positions_default = torch.arrange(max_seq_len, device=device)
        if theta != 0:
            i_vec = torch.arange(max_seq_len, device=device)[:, None]
            k_vec = torch.arange(d_k//2, device=device)[None, :]
            thetas = i_vec / theta ** (2*k_vec/d_k)
            # Typo in the assignment. There it says that k in {1, ..., d/2}.
            # Can either view as sin/cos or as complex
            R = torch.stack((thetas.cos(), thetas.sin()))
            # Complex version: 
            # R = torch.polar(torch.ones_like(thetas), thetas)
            R.to(device=device)
            self.register_buffer("R", R, persistent=False)



    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor: 
        # Basic version
        if self.theta == 0:
            return  x
    
        seq_len = x.shape[-2]
        if token_positions is None:
            even = x[...,::2]
            odd = x[...,1::2]
            c = self.R[0, :seq_len,...]
            s = self.R[1,:seq_len,...] 
            tmp = even * s + odd * c      
            x[...,::2] = even * c - odd *s
            x[...,1::2] = tmp
        else:
            even = x[...,token_positions,::2]
            odd = x[...,token_positions,1::2]
            c = self.R[0,token_positions, ...]
            s = self.R[1,token_positions,...] 
            tmp = even * s + odd * c      
            x[...,token_positions,::2] = even * c - odd *s
            x[...,token_positions,1::2] = tmp        
        return x

class Softmax(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim=dim


    def forward(self, x, dim = -1):
        self.dim = dim
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

def softmax(x, dim = -1):
        dim = dim
        orig_dtype = x.dtype

        x = x.float() # full precision
        # scale so that large value does not lead to inf value 
        max_val, _ = x.max(dim=dim, keepdim=True) # last dim max 
        
        # exp 
        x_scaled = torch.exp(x - max_val) # exp(x) 
        
        # sum of exp 
        x_sum = torch.sum(x_scaled, dim=dim, keepdim=True) # sum of exp(x) along last dim

        out = x_scaled / x_sum
        
        return out.to(orig_dtype) # exp(x) / sum(exp(x)), may be need broadcase

def scaled_dot_product_attention(Q:torch.Tensor, K: torch.Tensor, V, mask = None):
    d_k = Q.shape[-1]
    QKT = einops.einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    QKT.div_(np.sqrt(d_k))
    softmax_dim = len(QKT.shape) - 1
    seq_length = Q.shape[-2]
    if mask is not None:
        result = torch.where(mask[:seq_length,:seq_length],
        0,
        -float('inf'))
        A = softmax(QKT + result,dim = softmax_dim)
    else: 
        A = softmax(QKT,dim = softmax_dim)
    return einops.einsum(A, V, "... crud seq_length, ... seq_length d_v -> ... crud d_v")

class MultiHeadAttention(nn.Module):
    #    class multihead_self_attention(nn.Module): 
    def __init__(self, d_model:int, num_heads:int, max_seq_length:None, theta:None, device=None, dtype=None):
        super().__init__()

        self.W_QKV = Linear(d_model, 3*d_model, device=device, dtype=dtype)
        self.W_O = Linear(d_model, d_model,device=device, dtype=dtype)
        self.d_model = d_model
        self.num_heads = num_heads
        # if max_seq_length != None and theta != None:
        self.R = Rope(theta=theta, max_seq_len=max_seq_length, d_k=d_model//num_heads, device=device)
        self.cmask = torch.ones((max_seq_length,max_seq_length), dtype=torch.bool, device=device).tril()

    def forward(self, X:torch.tensor, token_positions = None):
        QKV = self.W_QKV.forward(X)
        QKV = einops.rearrange(QKV, "batch_size seq_length (three num_heads d_head) -> three num_heads batch_size seq_length d_head", three = 3, num_heads = self.num_heads)
        seq_length = QKV.shape[-2] # need to change to length of token positions
        QKV[:2, :] = self.R.forward(QKV[:2, :], token_positions=token_positions)
        # may need to squeeze here, not sure.
        # cmask = torch.ones((seq_length,seq_length), dtype=torch.bool).tril()
        A = scaled_dot_product_attention(QKV[0, :], QKV[1,:], QKV[2, :], mask=self.cmask)
        # print(A.shape, self.W_O.param.shape)
        A = einops.rearrange(A, "num_heads batch_size seq_length d_head -> batch_size seq_length (num_heads d_head)")
        out = self.W_O.forward(A)
        return out


class transformer_block(nn.Module): 
    def __init__(self, d_model:int, num_heads:int, d_ff:int = None, max_seq_length:int = None, theta:int = None, pre_RMS = True, post_RMS = False, activation = "", device=None, dtype=None):
        super().__init__()

        self.MHA = MultiHeadAttention(num_heads=num_heads, d_model=d_model, max_seq_length=max_seq_length, theta=theta, device=device, dtype=dtype)
        self.FFN = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, device=device)
        self.pre_RMS = pre_RMS
        self.post_RMS = post_RMS
        assert not (pre_RMS and post_RMS), "pre_RMS and post_RMS cannot both be True"
        self.RMSNorm1 = RMSnorm(d_modal=d_model, device=device, dtype=dtype)
        self.RMSNorm2 = RMSnorm(d_modal=d_model, device=device, dtype=dtype)


    def forward(self, X:torch.tensor):
        if self.pre_RMS:
            Y = X + self.MHA(self.RMSNorm1.forward(X))
            Z = Y + self.FFN.forward(self.RMSNorm2(Y))
            return Z
        if self.post_RMS:
            Y = self.RMSNorm1.forward(X + self.MHA(X))
            Z = self.RMSNorm2(Y + self.FFN(Y))
            return Z
        else:
            Y = X + self.MHA(X)
            Z = Y + self.FFN(Y)
            return Z
        
class transfoer_lm(nn.Module):
    def __init__(self, d_model:int, num_heads:int, vocab_size:int, context_length: int, num_layers: int, d_ff:int = None, theta:int = None, pre_RMS = True, post_RMS = False, activation = "", device=None, dtype=None):
        super().__init__()
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            TB = transformer_block(d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_seq_length=context_length, theta=theta, device=device,dtype=dtype, pre_RMS = pre_RMS, post_RMS = post_RMS, activation = activation)
            self.layers.append(TB)
        self.final_RMSNorm = RMSnorm(d_modal=d_model, device=device,dtype=dtype)
        self.output_layer = Linear(out_features=vocab_size, in_features=d_model, device=device,dtype=dtype)
        self.output_layer.weight.data.div_(np.sqrt(vocab_size + d_model)) # muP?
        self.pre_RMS =  pre_RMS
        self.post_RMS = post_RMS
    
    def forward(self, X:torch.Tensor):
        X = self.embedding.forward(X)
        for layer in self.layers:
            X = layer.forward(X)
        if self.pre_RMS or self.post_RMS:
            X = self.final_RMSNorm.forward(X)
        X = self.output_layer(X)
        return X
        
