from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
import numpy as np
import numpy.typing as npt

def get_batch(dataset:npt.NDArray,batch_size:int,context_length:int,device:str) -> tuple[torch.Tensor, torch.Tensor]:
    
    max_start=dataset.shape[0]-context_length-1
    starts=np.random.randint(0,max_start+1,size=batch_size)
    
    input_batch=np.stack([dataset[start:start+context_length] for start in starts])
    label_batch=np.stack([dataset[start+1:start+context_length+1] for start in starts])
    
    input_tensor=torch.tensor(input_batch,dtype=torch.long,device=device)
    label_tensor=torch.tensor(label_batch,dtype=torch.long,device=device)
    
    return input_tensor, label_tensor
    