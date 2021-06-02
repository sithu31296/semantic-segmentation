import torch
import numpy as np
import random

def fix_seeds(seed: int = 123) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def setup_cuda() -> None:
    torch.backends.cuda.deterministic = True
