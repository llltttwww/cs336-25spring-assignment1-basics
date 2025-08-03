from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
import numpy as np
import numpy.typing as npt


def save_checkpoint(model,optimizer,iteration,out):
    checkpoint={
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)
    
def load_checkpoint(src,model,optimizer):
    checkpoint=torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return checkpoint["iteration"]