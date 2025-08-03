import os
import math
import argparse
import wandb
import numpy as np
import torch
from typing import Optional

from .model import TransformerLM
from .dataset import get_batch
from .optimizer import Adamw, learning_rate_scheduler, gradient_clipping
from .serialization import save_checkpoint, load_checkpoint
from .utils import cross_entropy

from tqdm import trange
import time

def train(args):
    wandb.init(project=args.project, name=args.name, config=vars(args))
    train_dataset = np.load(args.train_dataset_path, mmap_mode='r')
    val_dataset=np.load(args.val_dataset_path, mmap_mode='r')
    print(f"Loaded dataset with {len(train_dataset)} tokens.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta
    ).to(device)

    optimizer = Adamw(
        model.parameters(), 
        lr=args.max_lr, 
        weight_decay=args.weight_decay,
        betas=args.betas, 
        eps=args.adamw_eps
    )
    
    loss_fn = cross_entropy

    # === logging setup ===
    start_time = time.time()
    total_params = sum(p.numel() for p in model.parameters())
    wandb.config.update({"num_parameters": total_params})
    print(f"Model has {total_params:,} parameters")

    t0 = time.time()  # for measuring batch_time

    for it in trange(args.num_iters):
        lr = learning_rate_scheduler(
            it, args.max_lr, args.min_lr, args.warmup_iters, args.cosine_iters
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        inputs, targets = get_batch(train_dataset, args.batch_size, args.context_length, device)
        logits = model(inputs)
        loss = loss_fn(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        if args.clip_grad_norm:
            gradient_clipping(model.parameters(), args.clip_grad_norm)
        optimizer.step()

        # === logging ===
        wandb.log({
            "iter": it,
            "loss": loss.item(),
            "lr": lr,
            "time_elapsed": time.time() - start_time,
            # Optional profiling:
            # "batch_time": time.time() - t0,
            # "gpu_mem": torch.cuda.max_memory_allocated() / 1024**2,
        }, step=it)

        # # === CLI logging ===
        # if it % args.log_interval == 0:
        #     print(f"[iter {it}] loss={loss.item():.4f} lr={lr:.6f}")
        t0 = time.time()

        # === Evaluation & checkpointing ===
        if it % args.eval_interval == 0 and it > 0:
            val_inputs, val_targets = get_batch(val_dataset, args.batch_size, args.context_length, device)
            with torch.no_grad():
                val_logits = model(val_inputs)
                val_loss = loss_fn(val_logits, val_targets)
            wandb.log({"val/loss": val_loss.item()}, step=it)

            save_path = os.path.join(args.output_dir, f"checkpoint_{it}.pt")
            save_checkpoint(model, optimizer, it, save_path)
        


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"  # Set to your desired GPU ID
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset_path", type=str, default="/workspace/home/luotianwei/cs336/assignment1-basics/data/TinyStoriesV2_GPT4-train.npy")
    parser.add_argument("--val_dataset_path", type=str, default="/workspace/home/luotianwei/cs336/assignment1-basics/data/TinyStoriesV2_GPT4-valid.npy")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/stantard_setting")
    parser.add_argument("--project", type=str, default="lm-train")
    parser.add_argument("--name", type=str, default="default_setting")
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate for cosine annealing")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--betas", type=tuple, default=(0.9, 0.95), help="Betas for AdamW optimizer")
    parser.add_argument("--adamw_eps", type=float, default=1e-8, help="Epsilon for AdamW optimizer")
    parser.add_argument("--clip_grad_norm", type=float, default=None)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--cosine_iters", type=int, default=40000)
    parser.add_argument("--num_iters", type=int, default=40000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=2000)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta",type=float,default=10000.0)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)