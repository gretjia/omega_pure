# OMEGA PURE: COSTLY DEBUG LESSONS

This document records the critical architectural and physical failure modes encountered during the transition from the legacy `omega` project to the `omega_pure` paradigm. **These lessons are written in blood (system crashes) and must be adhered to in all future iterations.**

## 1. The GPU vs. CPU Illusion (JAX XLA Death)
* **Symptom:** When running `omega_epiplexity_forge.py` (which uses JAX and `@jit`) on the AMD AI Max 395 128G Unified Memory nodes, the process completely hung, consuming endless hours per file without output.
* **Root Cause:** The nodes were operating in a CPU-only environment (`jax[cpu]`). JAX's XLA compiler is designed for static graph compilation on dedicated accelerators (TPU/GPU). When forced to compile an O(N^2) topological manifold calculation on a CPU with dynamically varying sequence lengths (due to different stock tick counts), XLA entered a catastrophic compilation loop.
* **The Fix:** We must **NEVER** use JAX `@jit` for dynamically sized loops in a CPU environment. We either fallback to pure `numpy` with Python `multiprocessing` for CPU nodes, or explicitly install `jax[rocm]` / `PyTorch ROCm` to unlock the actual AMD GPU (Radeon 8060S).

## 2. The Unified Memory Bios Trap
* **Symptom:** After successfully migrating to `PyTorch ROCm` on Linux (`linux1-lx`) to utilize the RDNA 3.5 GPU, the script crashed with a `HIP out of memory` error, stating it only had 60GB of capacity (despite the machine having 128GB of Unified RAM).
* **Root Cause:** The motherboard BIOS was configured to manually lock the UMA Frame Buffer Size (VRAM) to a specific number, capping the GPU's visibility to 60GB and preventing it from dynamically borrowing from the 128GB system pool.
* **The Fix:** The Chief Architect had to physically access the BIOS and set `UMA Frame Buffer Size` to **Auto**. This immediately expanded PyTorch's visible limit and allowed dynamic memory negotiation up to 96GB+.

## 3. The 1.2 Billion Row Death Spiral (OOM Livelock)
* **Symptom:** On extreme market days (e.g., `20240205` and `20240206` - the Quant crash, with over 1.2 billion ticks / 120M rows), the Linux node completely stopped responding to SSH (`Connection timed out`).
* **Root Cause:** The Linux OOM (Out Of Memory) Killer went rogue. Our initial data slicing method using Polars (`df.filter(pl.col("symbol") == sym)`) inside a Python for-loop of 7000 symbols caused Polars to perform a full 120M row table scan 7000 times ($O(N \cdot M)$ complexity). This created a "Memory Cloning Storm" that instantly consumed the 128GB RAM and 16GB Swap, forcing the OOM Killer to indiscriminately kill essential system services (including `tailscaled` and `sshd`).
* **The Fix:** 
  1. We must **NEVER** iterate and filter massive Polars DataFrames sequentially in Python.
  2. We pivoted to converting the Polars DataFrame to a zero-copy PyArrow-backed Pandas DataFrame: `df.to_pandas(use_pyarrow_extension_array=True)`.
  3. We then used Pandas' highly optimized C-level `.groupby("symbol")` to generate an iterative object.
  4. Added explicit `torch.cuda.empty_cache()` and `gc.collect()` every 1000 iterations to guarantee the VRAM and Host RAM never exceed safe boundaries.

## 4. Windows 11 WSL / DirectML Hostility
* **Symptom:** Attempting to harness the AMD GPU on the Windows node (`windows1-w1`) proved disastrous. `torch-directml` consumed 68GB of RAM instantly and deadlocked on basic matrix math. Attempting to install WSL2 to use native ROCm failed silently (exit code 1) despite enabling SVM in BIOS and forcing updates via MSI and Appx bundles.
* **Root Cause:** Windows 11's integration with DirectML for complex tensor operations is inefficient regarding memory mapping. Furthermore, WSL's background daemon often corrupts or deadlocks silently in customized or heavily security-patched Windows environments, making headless SSH deployment nearly impossible.
* **The Fix (The Talebian Way):** We abandoned the fragile Windows node for GPU computation. If the system is hostile, we do not fight it. We routed **100% of the workload** to the robust Linux node (`linux1-lx`), which processes the data at 1 minute/day. Windows is relegated to purely CPU-bound fallback tasks or cold storage.