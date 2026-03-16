# Linux1-lx Post-Mortem: UMA Fragmentation & I/O Livelock
**Date:** 2026-03-16

## The Problem
The `linux1-lx` node is repeatedly experiencing hard kernel crashes (requiring physical reboots) under both GPU and CPU workloads due to extreme memory fragmentation and allocation failures following a prior 1.2-Billion-Row OOM event.

## Hard Evidence
1.  **ROCm Segfaults (GPU):**
    *   **Trigger:** Executing PyTorch/JAX in conjunction with PyArrow batch reading.
    *   **Evidence:** `[dmesg] python3: segfault at 18 ip ... error 4 in libamdhip64.so` followed by `amdgpu: Freeing queue vital buffer, queue evicted`.
    *   **Meaning:** The AMD driver cannot handle simultaneous Host and Device memory mapping requests over the UMA (Unified Memory Architecture) without fatal page faults.

2.  **ZFS/Kernel I/O Livelock (CPU):**
    *   **Trigger:** Executing `pl.scan_parquet(...).collect()` on a 1.5GB (compressed) daily shard, attempting to load ~73 million rows into memory at once.
    *   **Evidence:** `[dmesg] exc_page_fault` followed by `Future hung task reports are suppressed`. The node completely drops off the LAN and Tailscale network.
    *   **Meaning:** The system memory manager (likely ZFS ARC cache) completely deadlocks when attempting to allocate the uncompressed data blocks, causing the OS network stack to hang indefinitely.

## The Mitigation (No Re-Format)
The hardware/drivers currently suffer from "large allocation allergy". To use this node without wiping the OS:
1.  **Abandon Global Loading:** `df.collect()` must be strictly forbidden in all future scripts.
2.  **Adopt Streaming Pipelining:** Data must be processed in micro-batches (e.g., `pyarrow.parquet.iter_batches(batch_size=1_000_000)`).
3.  **Explicit Garbage Collection:** Each micro-batch must be explicitly deleted (`del batch`, `gc.collect()`) before reading the next.