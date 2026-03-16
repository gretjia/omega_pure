# OMEGA PURE

This is the next-generation architecture for the OMEGA quantitative trading protocol, officially shifting from the legacy 1D Classical Sequence modeling (`/projects/omega`) into the **2D Spatio-Temporal Folding Paradigm**.

## 1. Core Philosophy (The Mathematical Canon)
As directed by the Chief Architect, OMEGA PURE rejects "dimensional arrogance". Market algorithms do not operate on a continuous 1D ticker tape; they operate across distinct days at identical intraday times. Thus, we fold `N` ticks into a `[Days, Ticks_per_day]` 2D tensor, allowing $O(1)$ Spatio-Temporal CNNs (ViT-MAE) to capture multi-day accumulation patterns without the $O(N^2)$ attention collapse of 1D Transformers.

While the *geometry* of how we look at the data has fundamentally changed to 2D, the underlying **Epistemic Trinity** (the mathematics) remains absolutely identical to the legacy system:
1.  **SRL (Square Root Law) Residual:** The physical footprint of momentum.
2.  **Epiplexity (MDL):** The measurement of information compression when algorithms force order.
3.  **Topology:** Geometric area of the price trajectory.

*Proof of Mathematical Viability:* A recent run of `tools/cloud_math_proof.py` across 188GB of faceless 1D data yielded an Asymmetry Ratio of **1.33**. The goal of the 2D shift is to push this Ratio beyond **2.0**.

---

## 2. Directory Structure & AI Auditor Guide

To all AI Agents, Auditors, and Executors: **Start your context-gathering in `/handover/` before touching any code.**

### `/handover/` (The Brain & Status)
This directory is the single source of truth for the active system state, blockers, and architectural mandates.
*   `LATEST.md`: The daily executive summary. What is currently running, what is blocked, and immediate next steps.
*   `PIPELINE_REBUILD_PLAN.md`: **[CRITICAL]** Read this to understand why we are currently running pure-CPU Numpy scripts instead of the GPU. It explains the "Symbol Paradox" and the AMD driver crash history.
*   `LINUX1_POST_MORTEM.md`: Detailed hardware-level diagnostics on why the legacy PyTorch/JAX `omega_tensor_materializer` scripts crash the `linux1-lx` kernel.

### Mathematical Core (The Physics & Truth)
These files contain the pure, hardware-agnostic implementations of the Epistemic Trinity (SRL, Epiplexity) and the loss functions. Modifying these alters the fundamental scientific hypothesis of the project.
*   **`tools/omega_tensor_materializer_numpy_streaming.py`**: **[ACTIVE]** Contains the active, stable `np_compute_srl_residual` and `np_compute_epiplexity` mathematical kernels currently running on the Windows node.
*   **`tools/cloud_math_proof.py`**: The raw mathematical verification engine used to prove the Asymmetry Ratio on GCP without ML models.
*   **`omega_2d_folded_mae.py`**: Defines the mathematical objective (FVU Loss) and the native 2D Spatio-Temporal topological structure (`SpatioTemporal2DMAE`).

### Engineering (ETL, Training & Backtesting)
These files handle data movement, model optimization, cloud execution, and simulated trading. They wrap the math core but do not define the physics.
*   **Data Rebuild & ETL:**
    *   **`tools/repack_to_ticker_shards.py`**: **[PENDING]** "The Chrono-Forge". The Architect's DuckDB Map-Reduce engine for spatial-temporal data dispersal.
*   **Cloud Training (Google Vizier & Vertex AI):**
    *   **`mac_blitz_v5_ignition.py`**: The master deployment script for the 100-node L4 GPU Wolfpack.
    *   **`vertex_mae_blitz_v5.py`**: The payload executed on Vertex AI for HPO.
    *   **`blitz_v5_hpo_config.yaml`**: The Google Vizier hyperparameter search space definitions (searching for `days` and `ticks_per_day`).
*   **Backtesting:**
    *   **`omega_parallel_crucible.py`**: The Talebian Oracle backtester. Embarrassingly parallel, isolated event-driven studies.

---

## 3. Deprecated Modules (Do Not Execute)
The following files have been moved to the `archive/` directory and are retained *only* for historical context. They suffer from fatal hardware incompatibilities (AMD ROCm segfaults) or architectural flaws (dropping `symbol` columns). **Do not use them to generate data:**
*   `archive/omega_tensor_materializer.py` (Legacy JAX, drops symbols, OOMs on large clusters)
*   `archive/omega_tensor_materializer_pytorch.py` (and its `_patched` variants - segfaults Linux Kernel via libamdhip64.so)
*   `archive/omega_epiplexity_forge.py` (and `_pytorch.py` - the underlying kernels causing the GPU crashes)