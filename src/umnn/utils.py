import torch
import math
from functools import lru_cache

@lru_cache(maxsize=32)
def compute_cc_weights(nb_steps, device="cpu"):
    """
    Computes Clenshaw-Curtis quadrature weights and steps.
    Cached to avoid recomputation for the same number of steps.
    """
    lam = torch.arange(0, nb_steps + 1, 1, device=device, dtype=torch.float32).reshape(-1, 1)
    lam = torch.cos((lam @ lam.T) * math.pi / nb_steps)
    lam[:, 0] = .5
    lam[:, -1] = .5 * lam[:, -1]
    lam = lam * 2 / nb_steps
    
    W = torch.arange(0, nb_steps + 1, 1, device=device, dtype=torch.float32).reshape(-1, 1)
    mask = (torch.arange(0, nb_steps + 1, device=device) % 2) != 0
    W[mask] = 0
    
    W = 2 / (1 - W ** 2)
    W[0] = 1
    W[mask] = 0
    
    cc_weights = torch.matmul(lam.T, W).view(-1)
    steps = torch.cos(torch.arange(0, nb_steps + 1, 1, device=device, dtype=torch.float32).reshape(-1, 1) * math.pi / nb_steps)
    
    return cc_weights, steps
