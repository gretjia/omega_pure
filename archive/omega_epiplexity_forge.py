"""
THE OMEGA PROTOCOL: THE DETERMINISTIC FORGE
Paradigm: Epiplexity & Kolmogorov Compression
Execution Target: AMD ROCm (JAX/XLA) via Unified Memory Architecture
Discipline: STRICT PURE FUNCTIONS. NO STATE. NO CLASSES. TENSOR IN -> TENSOR OUT.
"""

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

# =====================================================================
# MODULE 1: THE SQUARED ROOT LAW (SRL) FILTER
# Physics-based filtering to strip retail entropy and isolate 
# institutional liquidity boundaries.
# =====================================================================

@jit
def compute_srl_residual(
    price_change: jnp.ndarray, 
    order_flow: jnp.ndarray, 
    market_volume: jnp.ndarray, 
    volatility: jnp.ndarray, 
    gamma: float = 1.0
) -> jnp.ndarray:
    """
    [PURE FUNCTION]
    Computes the deviation from the theoretical Squared Root Law.
    Equation: Expected Impact = gamma * sigma * sgn(Q) * sqrt(|Q| / V)
    A high residual exposes a deterministic institutional footprint.
    """
    # Prevent division by zero using safe numerical boundaries
    safe_volume = jnp.maximum(market_volume, 1e-8)
    
    # Calculate physical limit of market impact
    normalized_flow = jnp.abs(order_flow) / safe_volume
    theoretical_impact = gamma * volatility * jnp.sign(order_flow) * jnp.sqrt(normalized_flow)
    
    # The pure mathematical footprint (Epiplexity precursor)
    residual = price_change - theoretical_impact
    return residual

# =====================================================================
# MODULE 2: HIGH-DIMENSIONAL TOPOLOGY (PHASE SPACE)
# Takens' Delay Embedding to reconstruct the hidden attractor.
# =====================================================================

@partial(jit, static_argnames=['dim', 'delay'])
def takens_embedding(ts: jnp.ndarray, dim: int = 10, delay: int = 1) -> jnp.ndarray:
    """
    [PURE FUNCTION]
    Transforms 1D time series residuals into a High-Dimensional Topological Manifold.
    Uses vectorized advanced indexing for zero-copy sliding windows.
    """
    valid_length = ts.shape[0] - (dim - 1) * delay
    
    starts = jnp.arange(valid_length)
    offsets = jnp.arange(dim) * delay
    
    # Broadcast to create the embedding matrix: shape (N, dim)
    indices = starts[:, None] + offsets[None, :]
    return ts[indices]

# =====================================================================
# MODULE 3: EPIPLEXITY METRIC (KOLMOGOROV COMPRESSIBILITY)
# =====================================================================

@jit
def compute_epiplexity_score(manifold: jnp.ndarray, epsilon: float = 0.05) -> jnp.ndarray:
    """
    [PURE FUNCTION]
    Computes Time-Bounded Entropy of the topological space (Correlation Sum / Betti-0 proxy).
    A sudden drop in entropy = Manifold Collapse = Highly Compressible Algorithmic Presence.
    """
    # Mathematical optimization to avoid O(N^2 * D) memory explosion
    # (a - b)^2 = a^2 + b^2 - 2ab
    sq_norms = jnp.sum(manifold**2, axis=1)
    sq_dists = sq_norms[:, None] + sq_norms[None, :] - 2.0 * jnp.dot(manifold, manifold.T)
    
    # Clip negative values that might arise from floating point inaccuracies
    sq_dists = jnp.maximum(sq_dists, 0.0)
    
    # Soft counting within epsilon radius (differentiable)
    connectivity = jax.nn.sigmoid((epsilon**2 - sq_dists) * 1000.0)
    
    # Local density estimation
    local_density = jnp.mean(connectivity, axis=1)
    safe_density = jnp.maximum(local_density, 1e-8)
    
    # Shannon entropy projection (Lower = More structured/compressible)
    entropy = -jnp.log(safe_density)
    return entropy

# =====================================================================
# MODULE 4: THE MASTER TENSOR FORGE
# =====================================================================

@partial(jit, static_argnames=['dim', 'delay'])
def forge_epiplexity_tensor(
    price_change: jnp.ndarray,
    order_flow: jnp.ndarray,
    market_volume: jnp.ndarray,
    volatility: jnp.ndarray,
    dim: int = 10,
    delay: int = 1,
    epsilon: float = 0.05
) -> jnp.ndarray:
    """
    [THE UNIVERSAL MACHINE]
    Ingests raw market entropy -> Outputs deterministic Epiplexity Tensor.
    """
    # 1. Physics Filter: Extract the non-random residual
    residuals = compute_srl_residual(price_change, order_flow, market_volume, volatility)
    
    # 2. Topology: Reconstruct the phase space
    manifold = takens_embedding(residuals, dim, delay)
    
    # 3. Compression: Measure topological entropy
    epiplexity = compute_epiplexity_score(manifold, epsilon)
    
    # 4. Temporal Alignment: Pad the initial missing values to match input length
    pad_len = (dim - 1) * delay
    padded_epiplexity = jnp.pad(epiplexity, (pad_len, 0), mode='edge')
    
    # Final Output Tensor Shape: (Time, 3) 
    # Columns: [Raw Price Change, SRL Residual, Epiplexity Compression Score]
    feature_tensor = jnp.stack([price_change, residuals, padded_epiplexity], axis=-1)
    
    return feature_tensor