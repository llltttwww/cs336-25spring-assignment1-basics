from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math




class Adamw(torch.optim.Optimizer):
    def __init__(self,
        params,
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    ):
        defaults = dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
        super().__init__(params, defaults)
        
    def step(self,closure:Optional[Callable]=None):
        loss=None if closure is None else closure()
        for group in self.param_groups:
            lr=group["lr"]
            weight_decay=group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                state=self.state[p]
                m=state.get("m", torch.zeros_like(p, memory_format=torch.preserve_format))
                v=state.get("v", torch.zeros_like(p, memory_format=torch.preserve_format))
                t=state.get("t",1)

                grad=p.grad
                m= beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad ** 2
                state["m"]=m
                state["v"]=v
                cur_lr=lr*math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                p.data-=cur_lr*(m/(v.sqrt()+eps))
                p.data-=lr*weight_decay*p.data
                
                state["t"]=t+1
                
                
                

def learning_rate_scheduler(it:int, max_learning_rate:int, min_learning_rate:int, warmup_iters: int, cosine_cycle_iters:int):
    if it<warmup_iters:
        return it/warmup_iters*max_learning_rate
    if it>=warmup_iters and it<= cosine_cycle_iters:
        return min_learning_rate + 0.5*(1+math.cos((it-warmup_iters)/(cosine_cycle_iters-warmup_iters)*math.pi))*(max_learning_rate-min_learning_rate)
    if it>cosine_cycle_iters:
        return min_learning_rate
        
        
def gradient_clipping(parameters:Iterable[torch.nn.Parameter],max_l2_norm:float) -> None:
    total_l2_norm=0
    eps=1e-6
    for p in parameters:
        if p.grad is not None:
            total_l2_norm+=(p.grad.data**2).sum().item()
    total_l2_norm=total_l2_norm**0.5
    if total_l2_norm>max_l2_norm:
        for p in parameters:
            if p.grad is not None:
                p.grad.data*=max_l2_norm/(total_l2_norm+eps)