import torch
import torch.nn as nn

def sparsemax(input, dim=0):
    """
    Sparsemax activation function.
    Implementation based on: https://arxiv.org/abs/1602.02068
    
    Sparsemax(z) = argmin_p ||p - z||^2 s.t. p in simplex.
    """
    # 1. Sort input in descending order
    sorted_z, _ = torch.sort(input, descending=True, dim=dim)
    
    # 2. Calculate cumulative sum
    z_cumsum = torch.cumsum(sorted_z, dim=dim)
    
    # 3. Find k(z)
    range_vec = torch.arange(1, input.size(dim) + 1, device=input.device).to(input.dtype)
    
    # Condition: 1 + k * z_{(k)} > sum_{j=1}^k z_{(j)}
    k_condition = 1 + range_vec * sorted_z > z_cumsum
    
    # Get the largest k that satisfies the condition
    # For a vector of size 3 (modality weights), k will be 1, 2, or 3.
    k_z = torch.sum(k_condition, dim=dim).long()
    
    # 4. Calculate threshold tau
    # We need to gather the correct cumulative sum at index k_z - 1
    # Since we are dealing with a small fixed dimension (3), we can simplify gather
    indices = (k_z - 1).view(-1)
    # Handling potential empty dimensions if input is higher than 1D
    # But for MoXGATE modality weights, input is always 1D (3,)
    if input.dim() == 1:
        tau_z = (z_cumsum[indices] - 1) / k_z
    else:
        # Batch support if needed in future
        tau_z = (z_cumsum.gather(dim, (k_z - 1).unsqueeze(dim)) - 1) / k_z.unsqueeze(dim)
        
    # 5. Result: max(0, z - tau)
    return torch.clamp(input - tau_z, min=0)

class Sparsemax(nn.Module):
    def __init__(self, dim=0):
        super(Sparsemax, self).__init__()
        self.dim = dim

    def forward(self, input):
        return sparsemax(input, dim=self.dim)
