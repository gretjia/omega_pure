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

### `/tools/` (ETL & Verification Pipeline)
Utility scripts for data reconstruction and mathematical validation.
*   **`tools/omega_tensor_materializer_numpy_streaming.py`**: **[ACTIVE]** The resilient, single-threaded, pure-Numpy, memory-safe data processor currently running on Windows to reconstruct the 188GB Base Matrix *with* the `symbol` and `time` columns intact.
*   **`tools/repack_to_ticker_shards.py`**: **[PENDING]** "The Chrono-Forge". The Architect's DuckDB Map-Reduce engine. It will scatter the output of the Numpy materializer into isolated, chronologically sorted ticker files (e.g., `000001.parquet`) ready for 2D ingestion.
*   `tools/cloud_math_proof.py`: The Vertex AI cloud script used to blindly verify the core equations without symbols.

### Root Directory (The 2D ML Pipeline)
*   **`omega_2d_folded_mae.py`**: The heart of the new architecture. Implements `SpatioTemporal2DMAE` (Native 2D CNN) and the `TimeFoldedDataset` (Zero-copy VRAM 1D->2D geometric folding).
*   **`omega_parallel_crucible.py`**: The Talebian Oracle backtester. Embarrassingly parallel, isolated event-driven studies.
*   **`blitz_v5_hpo_config.yaml`**: The Google Vizier hyperparameter search space definitions (searching for `days` and `ticks_per_day`, having abandoned 1D `stride`).
*   **`mac_blitz_v5_ignition.py`**: The master deployment script. Submits the 100-node L4 GPU Google Vizier HPO task.
*   `vertex_mae_blitz_v5.py`: The payload executed on Vertex AI by the ignition script.

---

## 3. Deprecated Modules (Do Not Execute)
The following files are retained *only* for historical context. They suffer from fatal hardware incompatibilities (AMD ROCm segfaults) or architectural flaws (dropping `symbol` columns). **Do not use them to generate data:**
*   `omega_tensor_materializer.py` (Legacy JAX, drops symbols, OOMs on large clusters)
*   `omega_tensor_materializer_pytorch.py` (and its `_patched` variants - segfaults Linux Kernel via libamdhip64.so)
*   `omega_epiplexity_forge.py` (and `_pytorch.py` - the underlying kernels causing the GPU crashes)