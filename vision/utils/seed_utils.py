# seed_utils.py
import random
import numpy as np
import torch

def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # More reproducible, slightly slower