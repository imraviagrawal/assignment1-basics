#!/usr/bin/env python3
"""
Transformer Debugging Interview — Qwen2.5‑style Tiny MoE Decoder
================================================================

Single-file script implementing a decoder-only Transformer with:
- RMSNorm + RoPE multi-head self-attention
- Mixture-of-Experts (top‑2) SwiGLU MLP
- Simple load-balancing auxiliary loss
- Tied token embeddings
- Tiny synthetic dataset and trainer

Run:
  python qwen25_moe_debug_interview.py

Notes:
  • The model is intentionally minimal and fast to train on CPU.
  • There is exactly one subtle bug somewhere in the model path. No hints.
"""

import math
import random
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Utilities
# -----------------------------

def seed_everything(seed: int = 2025):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Tiny dataset (character-level)
# -----------------------------

def make_tiny_corpus(n_repeat: int = 300) -> str:
    motifs = [
        "qwen moe tokens route to experts\n",
        "mixture of experts makes models scale\n",
        "routers balance loads in theory\n",
        "attention loves rotary embeddings\n",
    ]
    text = ("".join(motifs) * n_repeat)[:60000]
    return text


class CharTokenizer:
    def __init__(self, text: str):
        self.chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, s: str):
        return torch.tensor([self.stoi[c] for c in s], dtype=torch.long)

    def decode(self, ids):
        return "".join([self.itos[int(i)] for i in ids])


# -----------------------------
# Core layers
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * x * rms


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, seqlen: int, device=None, dtype=None) -> Tuple[torch.Tensor, torch.Tensor]:
        device = device or self.inv_freq.device
        dtype = dtype or self.inv_freq.dtype
        t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # [T, D/2]
        return freqs.cos().to(dtype), freqs.sin().to(dtype)

    @staticmethod
    def apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, H, T, D = x.shape
        x_ = x.view(B, H, T, D // 2, 2)
        x1, x2 = x_[..., 0], x_[..., 1]
        cos = cos.view(1, 1, T, D // 2)
        sin = sin.view(1, 1, T, D // 2)
        x1c = x1 * cos - x2 * sin
        x2c = x1 * sin + x2 * cos
        return torch.stack((x1c, x2c), dim=-1).view(B, H, T, D)


class MHA(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.hdim = dim // n_heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.hdim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.hdim).transpose(1, 2)  # (B,H,T,D)
        k = self.k(x).view(B, T, self.n_heads, self.hdim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_heads, self.hdim).transpose(1, 2)

        cos, sin = self.rope(T, device=x.device, dtype=x.dtype)
        q = RotaryEmbedding.apply(q, cos, sin)
        k = RotaryEmbedding.apply(k, cos, sin)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hdim)  # (B,H,T,T)
        att = att.masked_fill(attn_mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o(y)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_factor: float = 4.0):
        super().__init__()
        hidden = int(dim * hidden_factor)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Top2MoE(nn.Module):
    def __init__(self, dim: int, n_experts: int = 4, hidden_factor: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.n_experts = n_experts
        self.topk = 2
        self.experts = nn.ModuleList([SwiGLU(dim, hidden_factor) for _ in range(n_experts)])
        self.router = nn.Linear(dim, n_experts, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        tokens = x.reshape(B * T, C)
        logits = self.router(tokens)  # (N,E)
        # --------------------
        # gating
        probs = F.softmax(logits, dim=1)  # (N,E)
        # --------------------
        vals, idx = probs.topk(self.topk, dim=1)
        outs = torch.stack([expert(tokens) for expert in self.experts], dim=1)  # (N,E,C)
        picked = outs.gather(1, idx.unsqueeze(-1).expand(-1, self.topk, C))  # (N,2,C)
        mixed = (vals.unsqueeze(-1) * picked).sum(dim=1)  # (N,C)
        mixed = self.drop(mixed).view(B, T, C)

        # simple load-balancing aux loss (encourage uniform expert usage)
        usage = probs.mean(dim=0)  # (E,)
        aux_loss = (usage * usage).sum() * self.n_experts
        return mixed, aux_loss


class DecoderBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_experts: int, mlp_factor: float, dropout: float):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MHA(dim, n_heads, dropout)
        self.norm2 = RMSNorm(dim)
        self.moe = Top2MoE(dim, n_experts=n_experts, hidden_factor=mlp_factor, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x + self.drop(self.attn(self.norm1(x), attn_mask))
        moe_out, aux = self.moe(self.norm2(x))
        x = x + self.drop(moe_out)
        return x, aux


class TinyQwenMoE(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 192, n_layers: int = 4, n_heads: int = 6, n_experts: int = 4, mlp_factor: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            DecoderBlock(dim, n_heads, n_experts, mlp_factor, dropout) for _ in range(n_layers)
        ])
        self.norm_f = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # tie

    def forward(self, idx: torch.Tensor, attn_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embed(idx)
        aux_total = 0.0
        for blk in self.blocks:
            x, aux = blk(x, attn_mask)
            aux_total = aux_total + aux
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits, aux_total


# -----------------------------
# Data loader
# -----------------------------

@dataclass
class Config:
    dim: int = 192
    n_layers: int = 4
    n_heads: int = 6
    n_experts: int = 4
    mlp_factor: float = 4.0
    dropout: float = 0.0
    block_size: int = 128
    batch_size: int = 16
    lr: float = 3e-4
    max_steps: int = 500
    aux_coeff: float = 1e-2
    device: str = str(get_device())


class TinyLoader:
    def __init__(self, data: torch.Tensor, block_size: int, batch_size: int, device):
        self.data = data
        self.block = block_size
        self.batch = batch_size
        self.device = device

    def __iter__(self):
        return self

    def __next__(self):
        ix = torch.randint(0, len(self.data) - self.block - 1, (self.batch,))
        x = torch.stack([self.data[i : i + self.block] for i in ix])
        y = torch.stack([self.data[i + 1 : i + 1 + self.block] for i in ix])
        return x.to(self.device), y.to(self.device)


# -----------------------------
# Training
# -----------------------------

def causal_mask(T: int, device) -> torch.Tensor:
    m = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
    return m.view(1, 1, T, T)


def train():
    seed_everything()
    cfg = Config()
    device = torch.device(cfg.device)

    text = make_tiny_corpus(n_repeat=400)
    tok = CharTokenizer(text)
    data = tok.encode(text)

    loader = TinyLoader(data, cfg.block_size, cfg.batch_size, device)

    model = TinyQwenMoE(
        vocab_size=tok.vocab_size,
        dim=cfg.dim,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        n_experts=cfg.n_experts,
        mlp_factor=cfg.mlp_factor,
        dropout=cfg.dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.1)

    model.train()
    for step, (x, y) in enumerate(loader):
        if step >= cfg.max_steps:
            break
        T = x.size(1)
        mask = causal_mask(T, device)
        logits, aux = model(x, mask)
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss = ce + cfg.aux_coeff * aux

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % 50 == 0:
            print(f"step {step+1:4d} | loss {loss.item():.4f} | ce {ce.item():.4f} | aux {aux.item():.4f}")

    # simple sampling
    model.eval()
    with torch.no_grad():
        context = torch.tensor([[tok.stoi.get('q', 0)]], device=device)
        generated = [int(context.item())]
        T = cfg.block_size
        for _ in range(120):
            ctx = torch.tensor([generated[-T:]], device=device)
            mask = causal_mask(ctx.size(1), device)
            logits, _ = model(ctx, mask)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_id)
        print("\nSAMPLE:\n" + tok.decode(generated))


if __name__ == "__main__":
    train()
