# interview_ultrahard_llama_kv_buggy.py
# Ultra-hard interview: LLaMA-style tiny decoder (RMSNorm + SwiGLU + RoPE) with KV cache.
# There are deliberate bugs. Only the blocks containing bugs are named below.
# Your job: make all tests pass and the smoke training loss decrease.

import math, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------
# Config
# --------------------
seed = 777
random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)

VOCAB = 96
SEQ_LEN = 16
BATCH = 16

def make_batch(batch_size=BATCH, seq_len=SEQ_LEN):
    starts = torch.randint(0, VOCAB, (batch_size, 1))
    ar = torch.arange(seq_len).unsqueeze(0)
    seq = (starts + ar) % VOCAB
    return seq.long().to(device)

# --------------------
# RMSNorm (LLaMA style)
# --------------------
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps
    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight

# --------------------
# SwiGLU MLP
# --------------------
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

# --------------------
# RoPE utilities  (BUGGY BLOCK #1 is here)
# --------------------
def build_rope_cache(L, head_dim, base=10000.0, device="cpu", start=0):
    """
    Return cos, sin with absolute positions [start, start+L-1]
    shapes: (1,1,L,head_dim)
    """
    pos = (torch.arange(start, start + L, device=device).float()).unsqueeze(-1)  # (L,1)
    i = torch.arange(0, head_dim, 2, device=device).float()                      # (head_dim/2,)
    theta = 1.0 / (base ** (i / head_dim))                                       # (head_dim/2,)
    angles = pos * theta                                                         # (L, head_dim/2)

    cos = torch.zeros(L, head_dim, device=device)
    sin = torch.zeros(L, head_dim, device=device)
    # fill even/odd channels
    cos[:, 0::2] = torch.cos(angles)
    cos[:, 1::2] = torch.cos(angles)
    sin[:, 0::2] = torch.sin(angles)
    sin[:, 1::2] = torch.sin(angles)
    return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)

def apply_rope(x, cos, sin):
    """
    x:   (B, H, L, Dh)
    cos: (1, 1, L, Dh)  sin: same
    Returns rotated x.

    BUGGY BLOCK #1: this function contains a subtle rotation mistake.
    """
    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    # ----- INTENTIONAL BUG (signs/orientation) -----
    # Correct rotation should be:
    #   [x1', x2'] = [x1*cos - x2*sin, x1*sin + x2*cos]
    # The code below is subtly wrong but keeps shapes valid.
    xr1 = x1 * cos[..., ::2] - x2 * sin[..., ::2]   # <-- wrong (+ instead of -)
    xr2 = x1 * sin[..., ::2] + x2 * cos[..., ::2]  # <-- wrong sign pattern
    # -----------------------------------------------

    out = torch.zeros_like(x)
    out[..., ::2] = xr1
    out[..., 1::2] = xr2
    return out

# def apply_rope(x, cos, sin):
#     # x:   (B, H, L, Dh)
#     # cos/sin: (1, 1, L, Dh)
#     # Make sure dtypes match (important on amp/bfloat setups)
#     cos = cos.to(dtype=x.dtype)
#     sin = sin.to(dtype=x.dtype)

#     B, H, L, Dh = x.shape
#     # group last dim into pairs to avoid strided slicing writes
#     x_pair = x.view(B, H, L, Dh // 2, 2)      # (..., 2) = [x1, x2]
#     x1 = x_pair[..., 0]
#     x2 = x_pair[..., 1]

#     cos_pair = cos.view(1, 1, L, Dh // 2, 2)[..., 0]
#     sin_pair = sin.view(1, 1, L, Dh // 2, 2)[..., 0]

#     y1 = x1 * cos_pair - x2 * sin_pair
#     y2 = x1 * sin_pair + x2 * cos_pair

#     y = torch.stack([y1, y2], dim=-1).reshape(B, H, L, Dh)
#     return y

# --------------------
# Attention (BUGGY BLOCK #2 lives in the KV-cache path / masking)
# --------------------
class LlamaAttention(nn.Module):
    def __init__(self, d_model, n_heads, rope_base=10000.0, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.dh = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.rope_base = rope_base

    def forward(self, x, *, past_k=None, past_v=None, use_cache=False, return_attn=False):
        """
        x: (B, L, d_model)
        past_k, past_v: (B, H, Tpast, Dh) if provided
        Returns:
          y, new_k, new_v  (and optionally attn)
        """
        B, L, _ = x.shape
        H, Dh = self.n_heads, self.dh
        q = self.q_proj(x).view(B, L, H, Dh).transpose(1, 2)  # (B,H,L,Dh)
        k = self.k_proj(x).view(B, L, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, Dh).transpose(1, 2)

        past_len = 0 if past_k is None else past_k.size(2)

        # Build RoPE cache for *current tokens*
        # Note: For cached decoding, positions should start at `past_len`
        # BUGGY BLOCK #2: currently ignores offset when cache exists.
        cos, sin = build_rope_cache(L, Dh, base=self.rope_base, device=x.device, start=past_len)  # <-- should offset by past_len

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Concat KV for attention
        if past_k is not None:
            k_all = torch.cat([past_k, k], dim=2)  # (B,H,Tpast+L,Dh)
            v_all = torch.cat([past_v, v], dim=2)
        else:
            k_all, v_all = k, v

        # Build causal mask for (L, Tpast+L)
        Tk = k_all.size(2)
        i = torch.arange(L, device=x.device).unsqueeze(1)           # (L,1)
        j = torch.arange(Tk, device=x.device).unsqueeze(0)          # (1,Tk)
        mask = (j <= (i + past_len))                                # (L,Tk)   True=allowed
        mask = mask.unsqueeze(0).unsqueeze(0)                       # (1,1,L,Tk)

        scores = torch.matmul(q, k_all.transpose(-2, -1)) / math.sqrt(Dh)  # (B,H,L,Tk)
        scores = scores.masked_fill(~mask, float("-1e9"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)
        y = torch.matmul(attn, v_all)                               # (B,H,L,Dh)
        y = y.transpose(1, 2).contiguous().view(B, L, self.d_model)
        y = self.o_proj(y)

        if return_attn:
            return y, k_all, v_all, attn
        return y, k_all, v_all
# --------------------
# Block / Decoder
# --------------------
class LlamaBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.attn = LlamaAttention(d_model, n_heads, dropout=dropout)
        self.mlp = SwiGLU(d_model, d_ff)
        self.n1 = RMSNorm(d_model)
        self.n2 = RMSNorm(d_model)
    def forward(self, x, *, past=None, use_cache=False, return_attn=False):
        past_k, past_v = (None, None) if past is None else past
        if return_attn:
            y, k_all, v_all, attn = self.attn(self.n1(x), past_k=past_k, past_v=past_v, use_cache=use_cache, return_attn=True)
            x = x + y
            x = x + self.mlp(self.n2(x))
            return x, (k_all, v_all), attn
        else:
            y, k_all, v_all = self.attn(self.n1(x), past_k=past_k, past_v=past_v, use_cache=use_cache, return_attn=False)
            # else:
            x = x + y
            x = x + self.mlp(self.n2(x))
            return x, (k_all, v_all)

class TinyLlamaDecoder(nn.Module):
    def __init__(self, vocab, max_len, d_model=128, n_heads=4, n_layers=2, d_ff=256, dropout=0.0):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList([LlamaBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.nf = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.max_len = max_len
        nn.init.normal_(self.tok.weight, 0.0, 0.02)

    def forward(self, idx, *, past_kv=None, use_cache=False, return_attn=False):
        """
        idx: (B,L)
        past_kv: list of (k,v) or None, one per layer
        Returns:
          logits, new_kv (and optionally attn list)
        """
        B, L = idx.shape
        x = self.tok(idx)
        new_kv = []
        attns = []
        for li, blk in enumerate(self.blocks):
            past = None if past_kv is None else past_kv[li]
            if return_attn:
                x, kv, attn = blk(x, past=past, use_cache=use_cache, return_attn=True)
                attns.append(attn)
            else:
                x, kv = blk(x, past=past, use_cache=use_cache, return_attn=False)
            new_kv.append(kv)
        x = self.nf(x)
        logits = self.head(x)
        if return_attn:
            return logits, new_kv, attns
        return logits, new_kv

# --------------------
# Training utilities
# --------------------
criterion = nn.CrossEntropyLoss()

def prepare_io(batch):
    """
    Standard next-token language modeling I/O.
    """
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    return inputs, targets

def train_step(model, opt, batch):
    model.train()
    opt.zero_grad()
    inputs, targets = prepare_io(batch)
    logits, _ = model(inputs, use_cache=False)
    loss = criterion(logits.reshape(-1, VOCAB), targets.reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss.item()

def make_optimizer(model, lr=3e-4, wd=0.01):
    return torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9,0.95), eps=1e-8, weight_decay=wd)

# --------------------
# Tests
# --------------------
@torch.no_grad()
def rope_ref_test():
    """
    Compare attention matrices vs. a reference that applies *correct* RoPE to both q and k.
    """
    model = TinyLlamaDecoder(VOCAB, SEQ_LEN, d_model=128, n_heads=4, n_layers=1).to(device)
    x = make_batch(batch_size=1, seq_len=8)  # small
    emb = model.tok(x)  # (1,L,D)
    attn = model.blocks[0].attn

    B, L = 1, emb.size(1)
    H, Dh = attn.n_heads, attn.dh
    q = attn.q_proj(emb).view(B, L, H, Dh).transpose(1,2)
    k = attn.k_proj(emb).view(B, L, H, Dh).transpose(1,2)

    cos, sin = build_rope_cache(L, Dh, device=emb.device, start=0)
    # Reference rotation (correct)
    def _apply_correct(x, cos, sin):
        x1, x2 = x[..., ::2], x[..., 1::2]
        y1 = x1 * cos[..., ::2] - x2 * sin[..., ::2]
        y2 = x1 * sin[..., ::2] + x2 * cos[..., ::2]
        out = torch.zeros_like(x); out[..., ::2] = y1; out[..., 1::2] = y2
        return out
    q_ref = _apply_correct(q, cos, sin)
    k_ref = _apply_correct(k, cos, sin)

    scores_ref = torch.matmul(q_ref, k_ref.transpose(-2,-1)) / math.sqrt(Dh)
    mask = torch.tril(torch.ones(L, L, device=emb.device)).unsqueeze(0).unsqueeze(0)
    scores_ref = scores_ref.masked_fill(mask == 0, float("-1e9"))
    attn_ref = torch.softmax(scores_ref, dim=-1)

    # Module attention
    _y, _k, _v, attn_mod = attn(emb, use_cache=False, return_attn=True)

    diff = (attn_ref - attn_mod).abs().max().item()
    print("rope_ref_test | max diff:", diff)
    assert diff < 1e-5, "RoPE mismatch (q/k rotation incorrect)."

@torch.no_grad()
def kv_cache_consistency_test():
    """
    Compare final logits computed:
      (A) full forward on whole sequence vs.
      (B) autoregressive decode with KV-cache, one token at a time.
    They must match for the last position.
    """
    model = TinyLlamaDecoder(VOCAB, SEQ_LEN, d_model=128, n_heads=4, n_layers=2).to(device)
    seq = make_batch(batch_size=2, seq_len=SEQ_LEN)  # batch 2

    # A: full
    logits_full, _ = model(seq, use_cache=False)
    last_full = logits_full[:, -1, :]

    # B: incremental with cache
    past = None
    for t in range(SEQ_LEN):
        tok = seq[:, t:t+1]
        logits_step, past = model(tok, past_kv=past, use_cache=True)
    last_cache = logits_step[:, -1, :]

    diff = (last_full - last_cache).abs().max().item()
    print("kv_cache_consistency_test | max diff:", diff)
    assert diff < 1e-5, "KV-cache decoding not consistent with full pass."

# @torch.no_grad()
def grad_flow_test():
    """
    After one backward pass, key params must have non-zero grads.
    """
    model = TinyLlamaDecoder(VOCAB, SEQ_LEN, d_model=128, n_heads=4, n_layers=2).to(device)
    opt = make_optimizer(model)
    batch = make_batch(batch_size=4, seq_len=SEQ_LEN)
    model.train()
    inputs, targets = prepare_io(batch)
    logits, _ = model(inputs, use_cache=False)
    loss = criterion(logits.reshape(-1, VOCAB), targets.reshape(-1))
    loss.backward()
    grads = {name: (p.grad.abs().mean().item() if p.grad is not None else 0.0)
             for name, p in model.named_parameters()}
    print("grad_flow_test | q:", grads.get("blocks.0.attn.q_proj.weight", 0.0),
          "k:", grads.get("blocks.0.attn.k_proj.weight", 0.0),
          "v:", grads.get("blocks.0.attn.v_proj.weight", 0.0),
          "head:", grads.get("head.weight", 0.0))
    assert grads.get("blocks.0.attn.q_proj.weight", 0.0) > 0.0
    assert grads.get("blocks.0.attn.k_proj.weight", 0.0) > 0.0
    assert grads.get("blocks.0.attn.v_proj.weight", 0.0) > 0.0
    assert grads.get("head.weight", 0.0) > 0.0

def train_smoke_test():
    """
    Brief train should reduce loss.
    """
    model = TinyLlamaDecoder(VOCAB, SEQ_LEN, d_model=128, n_heads=4, n_layers=2).to(device)
    opt = make_optimizer(model, lr=3e-4, wd=0.01)
    N = 100
    losses = []
    for it in range(1, N+1):
        loss = train_step(model, opt, make_batch(BATCH, SEQ_LEN))
        losses.append(loss)
        if it % 20 == 0 or it == 1:
            print(f"iter {it:3d} | loss {loss:.4f}")
    print("train_smoke_test | first:", losses[0], " last:", losses[-1])
    assert losses[-1] < losses[0], "Loss did not decrease."

# --------------------
# Driver
# --------------------
if __name__ == "__main__":
    print("1) rope_ref_test (expected to FAIL while bug present)...")
    try:
        rope_ref_test()
        print("rope_ref_test passed (unexpected).")
    except AssertionError as e:
        print("rope_ref_test failed (expected). Error:", e)

    print("\n2) kv_cache_consistency_test (expected to FAIL while bug present)...")
    try:
        kv_cache_consistency_test()
        print("kv_cache_consistency_test passed (unexpected).")
    except AssertionError as e:
        print("kv_cache_consistency_test failed (expected). Error:", e)

    print("\n3) grad_flow_test (may pass/fail depending on your fixes)...")
    try:
        grad_flow_test()
        print("grad_flow_test passed.")
    except AssertionError as e:
        print("grad_flow_test failed. Error:", e)

    print("\n4) train_smoke_test (should FAIL until core bugs are fixed)...")
    try:
        train_smoke_test()
        print("train_smoke_test passed (unexpected).")
    except AssertionError as e:
        print("train_smoke_test failed (expected). Error:", e)
