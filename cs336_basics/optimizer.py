import os
import math 
import argparse
import torch
from einops import einsum, rearrange
import numpy as np 
from typing import IO, BinaryIO, Callable, Iterable, Optional, Union
from tqdm import tqdm, trange
from cs336_basics.nn import *


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-2, compile=False):
        if lr < 0 or betas[0] < 0 or betas[1] < 0 or weight_decay < 0 or eps < 0:
            raise ValueError(f"Invalid, negatove hyperparam")
        defaults = {'lr' : lr, 'betas' : betas, 'eps' : eps, 'lambda_wd' : weight_decay}
        super().__init__(params, defaults)
        
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group['betas']
            eps = group['eps']
            lambda_wd = group['lambda_wd']

            for p in group['params']:
                if p.grad is None:
                    continue
                    
                state = self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(p)
                if "v" not in state:
                    state["v"] = torch.zeros_like(p)
                if "t" not in state:
                    state["t"] = 1

                m = state["m"]
                v = state["v"]
                t = state["t"]

                grad = p.grad
                # m = beta1*m + (1-beta1)*grad # extra mem and not inplace, will be problematic when dealing with mem crunch
                m.mul_(beta1).add_(grad, alpha=1-beta1) # inplace and better mem 
                v.mul_(beta2).add_(grad.square(), alpha=1-beta2)
                # v = beta2*v + (1-beta2)*grad.square()

                alpha_t = lr*(math.sqrt(1 - math.pow(beta2, t)) / (1-math.pow(beta1, t)))
                
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                # updates
                denom = v.sqrt() + eps
                p.data.addcdiv_(m, denom, value=-alpha_t) # inplace update

                if lambda_wd != 0:
                    p.data.mul_(1 - lr*lambda_wd)
                # we update p.data not grad
                # grad = grad - alpha_t*(m/math.sqrt(v) + eps)
                # grad = grad - lr*lambda_wd*grad #weight decay
                state["m"] = m
                state["v"] = v
                state["t"] = t+1
        return loss

class scheduler():
    def __init__(self, optimizer, iter=0):
        self.optimizer = optimizer
        self.iter = iter

    def get_lr(self):
        raise NotImplemented
    
    def step(self):
        lr = self.get_lr()

        self.iter += 1

        for group in self.optimizer.param_groups:
            group["lr"] = lr


class cosine(scheduler):
    def __init__(self, optimizer, iter, max_lr, min_lr, warmup_end, cosine_end):
        super().__init__(optimizer, iter)
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_end = warmup_end
        self.cosine_end = cosine_end

    def get_lr(self):
        it = self.iter
        if it < self.warmup_end:
            return (it/max(self.warmup_end, 1))*self.max_lr
        
        elif self.warmup_end <= it <= self.cosine_end:
            return self.min_lr + 0.5*(1 + math.cos(((it - self.warmup_end)*math.pi)/(self.cosine_end - self.warmup_end)))*(self.max_lr - self.min_lr)
        else:
            return self.min_lr
        
def learning_rate_schedule(it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_final_iter: int):

    if it < warmup_iters: 
        return (it/warmup_iters)*max_learning_rate
    elif warmup_iters <= it <= cosine_cycle_final_iter:
        return min_learning_rate + .5*(1 + math.cos((it - warmup_iters)*math.pi/(cosine_cycle_final_iter - warmup_iters)))*(max_learning_rate-min_learning_rate)
    else:
        return min_learning_rate
    
# def gradient_clipping(parameters, max_l2_norm, eps = 1e-6):
#     # update the parameter where gradient is larger than norms 
#     for param in parameters:
#         if param.grad is None: 
#             continue
#         param_norms = param.grad.norm()
#         param.grad.mul_(max_l2_norm/(param_norms + eps))


def gradient_clipping(parameters, max_l2_norm, eps=1e-6):
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return 0.0

    # Compute total norm
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2) for p in params]), 2
    )

    # Only scale if norm too big
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for p in params:
            p.grad.mul_(scale)

    return total_norm.item()


def get_batch(data, batch_size, context_len, device=None):
    # data is len of token_ids
    # get the start tokens of size context_len 

    max_start = len(data) - context_len
    start_index = np.random.randint(0 , high=max_start, size=batch_size) # low, high and size 

    input = []
    targets = []
    # do we need to add start and end of index ? 
    for start in start_index:
        seq = data[start: start + context_len+1]
        input.append(seq[:-1])
        targets.append(seq[1:])
    
    return torch.tensor(input, dtype=torch.long, device=device), torch.tensor(targets, dtype=torch.long, device=device)
        

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'iteration': iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    return checkpoint['iteration']

def train(args):
    # todo 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load datasets using memory mapping
    train_data = np.load(args.train_data_path, mmap_mode='r')
    val_data = np.load(args.val_data_path, mmap_mode='r')

    # d_model, num_heads, vocab_size, context_length, num_layers, d_ff, theta, pre_RMS=True, post_RMS=False, activation='', device=None, dtype=None
    # load model 
    model = tranformer_lm(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.theta,
        device=device
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Load from checkpoint if available
    start_iter = 0
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        start_iter = load_checkpoint(args.checkpoint_path, model, optimizer)
        print(f"Resumed from checkpoint at iteration {start_iter}")

    progress_bar = trange(args.start_iter, args.total_iters, desc="Training")
    for t in progress_bar:
        # Learning rate schedule
        lr = learning_rate_schedule(t, args.lr, args.min_lr, args.warmup_iters, args.total_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Train step
        model.train()
        x, y = get_batch(train_data, args.batch_size, args.context_length, device)
        logits = model(x)
        loss = cross_entropy_loss(logits.view(-1, logits.size(-1)), y.view(-1))
        
        
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), args.max_grad_norm)
        optimizer.step()

        # Logging to progress bar
        if t % args.log_interval == 0:
            progress_bar.set_postfix(loss=loss.item(), lr=lr)

         # Evaluation
        if t % args.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                x_val, y_val = get_batch(val_data, args.batch_size, args.context_length, device)
                val_logits = model(x_val)
                val_loss = cross_entropy_loss(val_logits.view(-1, val_logits.size(-1)), y_val.view(-1))
                tqdm.write(f"[Eval @ Iter {t}] Val loss {val_loss.item():.4f}")

        # Checkpoint saving
        if args.checkpoint_path and t % args.ckpt_interval == 0:
            save_checkpoint(model, optimizer, t, args.checkpoint_path)
            tqdm.write(f"[Checkpoint @ Iter {t}] Saved to {args.checkpoint_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, required=True)
    parser.add_argument('--val_data_path', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, default="checkpoint.pt")

    parser.add_argument('--vocab_size', type=int, required=True)
    parser.add_argument('--context_length', type=int, default=32)

    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--theta', type=float, default=None)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_iters', type=int, default=200)
    parser.add_argument('--total_iters', type=int, default=5000)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--eval_interval', type=int, default=200)
    parser.add_argument('--ckpt_interval', type=int, default=500)

    args = parser.parse_args()
    train(args)