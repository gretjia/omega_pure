"""
THE OMEGA PROTOCOL: THE DETERMINISTIC FORGE (PyTorch / ROCm Backend)
Execution Target: AMD ROCm via PyTorch.
This explicitly circumvents JAX CPU deadlocks by utilizing the Radeon 8060S GPU.
"""

import torch
import torch.nn.functional as F

# =====================================================================
# MODULE 1: THE SQUARED ROOT LAW (SRL) FILTER
# =====================================================================

def compute_srl_residual(
    price_change: torch.Tensor, 
    order_flow: torch.Tensor, 
    market_volume: torch.Tensor, 
    volatility: torch.Tensor, 
    gamma: float = 1.0
) -> torch.Tensor:
    safe_volume = torch.clamp(market_volume, min=1e-8)
    normalized_flow = torch.abs(order_flow) / safe_volume
    theoretical_impact = gamma * volatility * torch.sign(order_flow) * torch.sqrt(normalized_flow)
    return price_change - theoretical_impact

# =====================================================================
# MODULE 2: HIGH-DIMENSIONAL TOPOLOGY (PHASE SPACE)
# =====================================================================

def takens_embedding(ts: torch.Tensor, dim: int = 10, delay: int = 1) -> torch.Tensor:
    valid_length = ts.shape[0] - (dim - 1) * delay
    
    # Use PyTorch Unfold (Zero-copy sliding window)
    # Shape: (N, 1) -> (valid_length, dim)
    if delay == 1:
        return ts.unfold(0, dim, 1)
    else:
        # For delay > 1, we construct the indices manually
        starts = torch.arange(valid_length, device=ts.device)
        offsets = torch.arange(dim, device=ts.device) * delay
        indices = starts[:, None] + offsets[None, :]
        return ts[indices]

# =====================================================================
# MODULE 3: EPIPLEXITY METRIC (KOLMOGOROV COMPRESSIBILITY)
# =====================================================================

def compute_epiplexity_score(manifold: torch.Tensor, epsilon: float = 0.05) -> torch.Tensor:
    # PyTorch highly optimized pairwise distance (cdist handles this mathematically)
    # P = cdists(A, B, p=2) -> computes Euclidean distance. We square it for sq_dists.
    sq_dists = torch.cdist(manifold, manifold, p=2.0) ** 2
    
    # Soft counting
    connectivity = torch.sigmoid((epsilon**2 - sq_dists) * 1000.0)
    
    # Local density estimation
    local_density = torch.mean(connectivity, dim=1)
    safe_density = torch.clamp(local_density, min=1e-8)
    
    # Shannon entropy projection
    entropy = -torch.log(safe_density)
    return entropy

# =====================================================================
# MODULE 4: THE MASTER TENSOR FORGE
# =====================================================================

def forge_epiplexity_tensor(
    price_change: torch.Tensor,
    order_flow: torch.Tensor,
    market_volume: torch.Tensor,
    volatility: torch.Tensor,
    dim: int = 10,
    delay: int = 1,
    epsilon: float = 0.05
) -> torch.Tensor:
    """
    Ingests raw market entropy -> Outputs deterministic Epiplexity Tensor.
    """
    residuals = compute_srl_residual(price_change, order_flow, market_volume, volatility)
    manifold = takens_embedding(residuals, dim, delay)
    epiplexity = compute_epiplexity_score(manifold, epsilon)
    
    pad_len = (dim - 1) * delay
    padded_epiplexity = F.pad(epiplexity, (pad_len, 0), mode='constant', value=0.0)
    
    return torch.stack([price_change, residuals, padded_epiplexity], dim=-1)