"""
Qwen-style Transformer implemented from scratch in PyTorch.

Core components included:
- Decoder-only, dense Transformer
- RMSNorm
- Rotary Position Embeddings (RoPE) w/ optional scaling (extrapolation)
- Multi-Head Self-Attention w/ optional Grouped-Query Attention (num_kv_heads)
- Optional Q/K/V learned biases
- SwiGLU MLP
- Causal masking & kv-cache for fast generation
- Weight tying between token embedding and lm_head

This implementation is educational and aims for clarity over peak speed.
Tested on PyTorch 2.2+.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# Config
# =========================

@dataclass
class QwenConfig:
    vocab_size: int = 151936            # can set to your tokenizer size; placeholder default
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_kv_heads: int = 32              # = num_attention_heads => MHA; < num_heads => GQA
    intermediate_size: int = 11008      # typical ~ 2.7x hidden for SwiGLU
    max_position_embeddings: int = 8192 # training context; RoPE lets you extrapolate with scaling
    rope_theta: float = 10000.0
    rope_scaling: Optional[Tuple[str, float]] = None  # ("linear", factor) or None
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    emb_dropout: float = 0.0
    layer_norm_eps: float = 1e-6        # RMSNorm epsilon
    use_qkv_bias: bool = True
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02
    dtype: torch.dtype = torch.bfloat16 # set to bfloat16 for speed on modern GPUs; fall back to float32
    device: str = "cpu"

    def __post_init__(self):
        assert self.hidden_size % self.num_attention_heads == 0, "hidden must be divisible by heads"
        assert self.num_attention_heads % self.num_kv_heads == 0, "heads must be divisible by kv_heads"

# =========================
# Utils: RMSNorm
# =========================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x

# =========================
# Utils: Rotary Positional Embeddings (RoPE)
# =========================

class RotaryEmbedding(nn.Module):
    """Applies RoPE to query/key. Supports optional linear scaling for extrapolation.

    Reference: RoPE (Su et al., 2021). Implementation mirrors llama/qwen style.
    """
    def __init__(self, dim: int, base: float = 10000.0, max_position: int = 8192,
                 scaling: Optional[Tuple[str, float]] = None, device: Optional[str] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position = max_position
        self.device = device
        self.dtype = dtype
        self.scaling = scaling
        # Precompute inv freqs
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rope_angles(self, seq_len: int) -> torch.Tensor:
        if self.scaling is None:
            t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
        else:
            kind, factor = self.scaling
            if kind != "linear":
                raise ValueError("Only linear rope scaling is implemented")
            t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32) / factor
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)  # (T, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (T, dim)
        cos = emb.cos()[None, None, :, :]  # (1,1,T,dim)
        sin = emb.sin()[None, None, :, :]
        return cos.to(dtype=self.dtype), sin.to(dtype=self.dtype)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., :x.shape[-1]//2]
        x2 = x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: (B, n_h, T, h_d)
        return (x * cos[..., :x.size(-1)]) + (RotaryEmbedding._rotate_half(x) * sin[..., :x.size(-1)])

# =========================
# Attention
# =========================

class QwenAttention(nn.Module):
    def __init__(self, cfg: QwenConfig):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=cfg.use_qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=cfg.use_qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=cfg.use_qkv_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(cfg.attn_dropout)

        # RoPE for q/k
        self.rope = RotaryEmbedding(
            dim=self.head_dim,
            base=cfg.rope_theta,
            max_position=cfg.max_position_embeddings,
            scaling=cfg.rope_scaling,
            device=cfg.device,
            dtype=cfg.dtype,
        )

    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        # x: (B, kv_heads, T, h_d) -> (B, heads, T, h_d)
        if n_rep == 1:
            return x
        b, kvh, t, d = x.shape
        x = x[:, :, None, :, :].expand(b, kvh, n_rep, t, d)
        return x.reshape(b, kvh * n_rep, t, d)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = x.shape
        H, HKV, D = self.num_heads, self.num_kv_heads, self.head_dim

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, H, D).transpose(1, 2)  # (B,H,T,D)
        k = k.view(B, T, HKV, D).transpose(1, 2)  # (B,HKV,T,D)
        v = v.view(B, T, HKV, D).transpose(1, 2)  # (B,HKV,T,D)

        # Apply RoPE to q,k
        cos, sin = self.rope._rope_angles(T if kv_cache is None else kv_cache[0].size(2) + T)
        # Take only the last T positions for the current chunk
        cos_cur = cos[..., -T:, :]
        sin_cur = sin[..., -T:, :]
        q = self.rope.apply_rotary(q, cos_cur, sin_cur)
        k = self.rope.apply_rotary(k, cos_cur, sin_cur)  # apply with same positions for simplicity in chunked gen

        # Append to cache if provided
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        new_cache = (k, v)

        # Expand kv to full heads if using GQA
        n_rep = H // HKV
        k_full = self._repeat_kv(k, n_rep)
        v_full = self._repeat_kv(v, n_rep)

        # Attention
        attn_scores = torch.matmul(q * self.scale, k_full.transpose(-2, -1))  # (B,H,T,TL)
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask  # mask should be broadcastable to (B,1,T,TL)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        out = torch.matmul(attn_probs, v_full)  # (B,H,T,D)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.o_proj(out)
        return out, new_cache

# =========================
# MLP (SwiGLU)
# =========================

class SwiGLU(nn.Module):
    def __init__(self, cfg: QwenConfig):
        super().__init__()
        self.up = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.gate = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))

# =========================
# Transformer Block
# =========================

class QwenBlock(nn.Module):
    def __init__(self, cfg: QwenConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.attn = QwenAttention(cfg)
        self.ln2 = RMSNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.mlp = SwiGLU(cfg)
        self.resid_dropout = nn.Dropout(cfg.resid_dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        a, new_cache = self.attn(self.ln1(x), attn_mask=attn_mask, kv_cache=kv_cache)
        x = x + self.resid_dropout(a)
        m = self.mlp(self.ln2(x))
        x = x + self.resid_dropout(m)
        return x, new_cache

# =========================
# Model
# =========================

class QwenModel(nn.Module):
    def __init__(self, cfg: QwenConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.drop = nn.Dropout(cfg.emb_dropout)
        self.blocks = nn.ModuleList([QwenBlock(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)
        self.to(device=cfg.device, dtype=cfg.dtype)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.initializer_range)

    @staticmethod
    def causal_mask(device: torch.device, dtype: torch.dtype, q_len: int, kv_len: int) -> torch.Tensor:
        # (1,1,q_len,kv_len) with -inf above diagonal
        mask = torch.full((1, 1, q_len, kv_len), float('-inf'), device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=1 + (kv_len - q_len))
        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass returning logits and updated caches.

        input_ids: (B, T)
        kv_caches: list per layer of tuples (k, v) with shapes:
            k: (B, kv_heads, T_total, head_dim)
            v: (B, kv_heads, T_total, head_dim)
        """
        B, T = input_ids.shape
        device = input_ids.device
        x = self.embed_tokens(input_ids)
        x = self.drop(x)

        new_caches: List[Tuple[torch.Tensor, torch.Tensor]] = []

        # Determine kv length for mask
        if kv_caches is None:
            kv_caches = [None] * len(self.blocks)
            kv_len = T
        else:
            kv_len = kv_caches[0][0].size(2) + T if kv_caches[0] is not None else T

        attn_mask = self.causal_mask(device, dtype=torch.float32, q_len=T, kv_len=kv_len)

        for blk, cache in zip(self.blocks, kv_caches):
            x, new_cache = blk(x, attn_mask=attn_mask, kv_cache=cache)
            new_caches.append(new_cache)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, new_caches

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        B, T = input_ids.shape
        device = input_ids.device
        kv_caches: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * len(self.blocks)
        out_ids = input_ids
        for _ in range(max_new_tokens):
            logits, kv_caches = self.forward(out_ids[:, -1:].contiguous() if out_ids.shape[1] > 1 else out_ids, kv_caches)
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            out_ids = torch.cat([out_ids, next_id], dim=1)
            if eos_token_id is not None and (next_id == eos_token_id).all():
                break
        return out_ids

# =========================
# Example usage (toy)
# =========================
if __name__ == "__main__":
    # Tiny config for sanity test on CPU
    cfg = QwenConfig(
        vocab_size=32000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_kv_heads=4,  # GQA
        intermediate_size=768,
        max_position_embeddings=1024,
        dtype=torch.float32,
        device="cpu",
    )

    model = QwenModel(cfg)
    B, T = 2, 16
    x = torch.randint(0, cfg.vocab_size, (B, T))
    logits, caches = model(x)
    print("logits:", logits.shape)

    # Autoregressive generation (random tokens as prompt)
    prompt = torch.randint(0, cfg.vocab_size, (1, 8))
    y = model.generate(prompt, max_new_tokens=10, temperature=0.8, top_k=50)
    print("generated shape:", y.shape)
