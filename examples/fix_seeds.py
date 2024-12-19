import random
import numpy as np
import torch

def fix_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
def set_device(cuda):
    device = torch.device(
        f"cuda:{cuda}"
        if torch.cuda.is_available() and cuda >= 0
        else None
    )
    return device