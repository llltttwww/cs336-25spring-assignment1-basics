from .model import TransformerLM
import torch
import torch.nn as nn
from torch import Tensor
from einops import einsum, rearrange
from jaxtyping import Int, Float, Bool
from typing import Optional

def cross_entropy(logits: Float[Tensor,"batch seq_len vocab_size"], targets: Int[Tensor, "batch seq_len"]) -> Float[Tensor, "(,)"]:
    useful_logits=torch.gather(logits,dim=-1 ,index=targets.unsqueeze(-1)).squeeze(-1)
    max_logits=torch.max(logits,dim=-1,keepdim=False).values
    probs_stable=useful_logits-max_logits-torch.log(torch.sum(torch.exp(logits-max_logits.unsqueeze(-1)),dim=-1))
    
    return -probs_stable.mean()


def print_parameter_count_by_module(model: nn.Module):
    """
    Print the number of trainable parameters in each named module of the model.

    Args:
        model (nn.Module): The model to analyze.
    """
    from collections import defaultdict

    param_count = defaultdict(int)

    for name, param in model.named_parameters():
        if param.requires_grad:
            # 获取模块前缀（例如 "layers.0.attn.q_proj.weight" -> "layers.0.attn.q_proj"）
            module_name = ".".join(name.split(".")[:-1])
            param_count[module_name] += param.numel()

    total = 0
    print(f"{'Module':50} | {'# Params':>10}")
    print("-" * 65)
    for module, count in sorted(param_count.items(), key=lambda x: -x[1]):
        print(f"{module:50} | {count:10,}")
        total += count
    print("-" * 65)
    print(f"{'Total':50} | {total:10,}")


if __name__=='__main__':
    model = TransformerLM(vocab_size=50257, context_length=1024, d_model=1600, num_layers=48, num_heads=25, d_ff=6400, rope_theta=10000.0)
    total_params = print_parameter_count_by_module(model)
    print(f"Total parameters in the model: {total_params}")