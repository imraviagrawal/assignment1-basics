import math 
import torch
from einops import einsum, rearrange
import numpy as np 
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