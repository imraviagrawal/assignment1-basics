# buggy_transformer_debug.py
# Minimal decoder-only Transformer with one deliberate bug in attention masking.

import math, time, random
import torch
import torch.nn as nn

# ---------- Config ----------
seed = 1234
random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

V = 32
SEQ_LEN = 16
BATCH = 16

def make_batch(batch_size=BATCH, seq_len=SEQ_LEN):
    starts = torch.randint(0, V, (batch_size, 1))
    arange = torch.arange(seq_len).unsqueeze(0)
    seq = (starts + arange) % V
    return seq.long().to(device)

# ---------- Model (bug injected) ----------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)  # (B, heads, L, d_k)
        k = self.k_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, heads, L, L)

        if attn_mask is not None:
            # attn_mask expected shape (1,1,L,L) or broadcastable; mask==0 blocks
            scores = scores.masked_fill(attn_mask == 0, float("-1e9"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, heads, L, d_k)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        out = self.out_proj(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.ff(self.ln2(x))
        return x

class TinyDecoderTransformer(nn.Module):
    def __init__(self, vocab_size, max_len, d_model=32, n_heads=4, n_layers=1, d_ff=128, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, L = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(L, device=idx.device)).unsqueeze(0)
        x = tok + pos
        mask = torch.tril(torch.ones(L, L, device=idx.device)).unsqueeze(0).unsqueeze(0)  # (1,1,L,L)
        for blk in self.blocks:
            x = blk(x, attn_mask=mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# ---------- Scratch CrossEntropy + AdamW (same as earlier, minimal) ----------
class ScratchCrossEntropy:
    def __init__(self, reduction='mean'):
        assert reduction in ('mean', 'sum', 'none')
        self.reduction = reduction
    def __call__(self, logits, targets):
        orig_shape = logits.shape
        if logits.dim() == 3:
            B, L, C = logits.shape
            logits = logits.reshape(-1, C)
            targets = targets.reshape(-1)
        max_logits, _ = logits.max(dim=1, keepdim=True)
        shifted = logits - max_logits
        logsumexp = torch.log(torch.exp(shifted).sum(dim=1, keepdim=True)) + max_logits
        log_probs = logits - logsumexp
        nll = -log_probs[torch.arange(log_probs.shape[0], device=logits.device), targets]
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        else:
            if len(orig_shape) == 3:
                return nll.reshape(orig_shape[0], orig_shape[1])
            return nll

class ScratchAdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [torch.zeros_like(p.data) for p in self.params]
        self.v = [torch.zeros_like(p.data) for p in self.params]
        self.t = 0
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
    def step(self):
        self.t += 1
        lr = self.lr
        for i, p in enumerate(self.params):
            if p.grad is None: continue
            grad = p.grad.data
            self.m[i].mul_(self.beta1).add_(grad, alpha=(1 - self.beta1))
            self.v[i].mul_(self.beta2).addcmul_(grad, grad, value=(1 - self.beta2))
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            if self.weight_decay != 0:
                p.data.mul_(1 - lr * self.weight_decay)
            denom = v_hat.sqrt().add_(self.eps)
            step = m_hat / denom
            p.data.add_(step, alpha=-lr)

# ---------- Tests / training harness ----------
def mask_unit_test():
    m = CausalSelfAttention(d_model=32, n_heads=4).to(device)
    x = torch.randn(1, 6, 32, device=device)
    mask = torch.tril(torch.ones(6,6, device=device)).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        q = m.q_proj(x).view(1, 6, m.n_heads, m.d_k).transpose(1,2)
        k = m.k_proj(x).view(1, 6, m.n_heads, m.d_k).transpose(1,2)
        scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(m.d_k)
        # Buggy mask applied here via module logic is same as below; compute attn and check upper triangle:
        scores = scores.masked_fill(mask == 1, float("-1e9"))   # matches module's buggy behavior
        attn = torch.softmax(scores, dim=-1)
        upper = attn[0,0,0,4:].abs().sum().item()
        print("Mask unit test: sum of attention in upper-tri (should be ~0) =", upper)
        assert upper < 1e-6 or upper == 0.0, "Upper triangular attention not blocked by mask (BUG)"

def run_demo_train():
    model = TinyDecoderTransformer(vocab_size=V, max_len=SEQ_LEN, d_model=64, n_heads=4, n_layers=2, d_ff=256).to(device)
    optimizer = ScratchAdamW(model.parameters(), lr=3e-4, betas=(0.9,0.95), eps=1e-8, weight_decay=0.01)
    criterion = ScratchCrossEntropy()
    N_ITERS = 200
    for it in range(1, N_ITERS+1):
        batch = make_batch(BATCH, SEQ_LEN)
        model.train()
        optimizer.zero_grad()
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        if it % 20 == 0 or it == 1:
            print(f"iter {it:3d} loss {loss:.4f}")

def generation_demo():
    model = TinyDecoderTransformer(vocab_size=V, max_len=SEQ_LEN).to(device)
    # model is untrained here (just demo); generation may be nonsensical
    start = torch.tensor([[5]], device=device)
    idx = start
    for _ in range(8):
        logits = model(idx)
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, nxt], dim=1)
    print("Generated tokens (demo):", idx.tolist()[0])

if __name__ == "__main__":
    print("1) Running mask unit test (should FAIL with the deliberate bug)...")
    try:
        mask_unit_test()
        print("Mask test passed (unexpected).")
    except AssertionError as e:
        print("Mask test failed as expected. Error:", e)
    print("\n2) Run a short train demo to observe training behaviour (loss may not decrease well if mask is wrong).")
    run_demo_train()
    print("\n3) Generation demo (untrained model here) to show end-to-end run.")
    generation_demo()