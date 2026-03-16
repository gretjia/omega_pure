# OMEGA PURE: THE 2D DATA REBUILD PIPELINE (The "Symbol" Paradox)
**Last Updated: 2026-03-16**

## 1. The Core Blocker (The Symbol Paradox)
* **The Issue:** The 188GB `base_matrix_shards` data currently on GCP (and previously generated locally) is fundamentally **incompatible** with the new 2D Time-Folded MAE architecture.
* **The Reason:** The old `omega_tensor_materializer.py` (designed for 1D Transformer paper-tape inputs) explicitly stripped the `symbol` and `time` columns before saving the Parquet files to save space.
* **The Result:** Without the `symbol` column, the Chief Architect's `The Chrono-Forge` (DuckDB repacker) instantly fails with `[FATAL] Could not auto-detect Ticker column.` It cannot magically guess which row belongs to which stock.

## 2. The Unrecoverable Hardware State
* **The Illusion of GPU:** Attempts to use PyTorch ROCm on `linux1-lx` to regenerate the data instantly result in `segfault at 18 in libamdhip64.so` causing kernel panics. The AMD driver's UMA (Unified Memory Architecture) is in a completely broken state following a previous 1.2-Billion-Row OOM crash.
* **The Livelock:** Attempts to use Polars or PyArrow to read large files natively on `linux1-lx` currently cause severe I/O livelocks and SSH disconnects, indicating deeper ZFS or OS-level instability.

## 3. The Rebuild Pipeline (Two-Step Execution)
To fix this and enable 2D training, we MUST execute a strict two-step pipeline utilizing stable, pure-CPU architecture.

### STEP 1: The Origin Forge (Recovering the Symbol)
* **Goal:** Regenerate the 188GB matrix from the 2.2TB `latest_base_l1` raw data lake, but this time **preserve the `symbol` and `time` columns**.
* **Tool:** `omega_tensor_materializer_numpy.py` (A purely CPU-bound, vectorized Numpy script specifically written to avoid PyTorch/JAX/Polars segfaults).
* **Execution:** Run simultaneously on `linux1-lx` (552 files) and `windows1-w1` (191 files) to bypass hardware fragility.
* **ETA:** ~10-14 hours (CPU-bound constraint).

### STEP 2: The Chrono-Forge (Spatial-Temporal Dispersal)
* **Goal:** Take the newly generated, symbol-aware 188GB data and use DuckDB to scatter it into 5000 individual, strictly time-sorted ticker files (e.g., `000001.parquet`).
* **Tool:** `tools/repack_to_ticker_shards.py` (The Architect's DuckDB Map-Reduce payload).
* **Execution:** Once Step 1 finishes, this will run locally using 100GB strict memory limits.
* **ETA:** ~30-45 minutes (I/O-bound speed).

## 4. Final HPO Ignition
Only after Step 2 completes and the pure Ticker Shards are uploaded to `gs://omega-pure-data/ticker_matrix_shards_v5/` can the 100x L4 Vertex AI Blitzkrieg (Phase 2) be successfully ignited.