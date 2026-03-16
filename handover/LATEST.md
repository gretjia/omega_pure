# Omega Pure - Project LATEST Handover State
Last Updated: 2026-03-15 (Sunday)

## 1. STRATEGIC PIVOT: 2D Time-Folded MAE
*   **Status:** Abandoned 1D Transformer approach. Shifted to Native 2D CNN (ViT-MAE for finance) based on the "Space-bounded Entropy / Finite Window Theory".
*   **Fatal Discovery (1D):** Feeding A-share data as a 1D paper tape caused "graph bandwidth tearing" (4800 ticks separating adjacent daily closing times) and massive $O(N^2)$ memory explosions.
*   **The Cure (2D):** Implemented `SpatioTemporal2DMAE` and `TimeFoldedDataset` in `omega_2d_folded_mae.py` to instantly fold 1D ticker data into `[Days, Ticks_per_day]` 2D matrices in GPU VRAM (zero-copy), reducing attention complexity to $O(1)$ locally.

## 2. KEY TECHNICAL MILESTONE: Phase 2 HPO Re-Ignition
*   **Target:** `blitz_v5_hpo_config.yaml` and `vertex_mae_blitz_v5.py`
*   **Discipline (No Hard Encoding):** We no longer guess the window size. We use Google Vizier to search for the physical constants of institutional "invisibility cloaks" (Algorithms like VWAP/TWAP).
*   **Search Space:**
    *   `days` (Macro accumulation cycle): `[3, 5, 8, 13, 21]`
    *   `ticks_per_day` (Intraday resolution): `[16, 32, 64, 128]` (Matches 1.8min to 15min real execution slices).

## 3. REAL QUOTA AUDIT (Verified via API)
*   **Non-Preemptible L4:** 100 units (Confirmed).
*   **Standard CPU (us-central1):** 400 vCPUs. (Recent request to increase from 400 to 500 was **Denied**).
*   **Standard CPU (global):** 500 vCPUs (Approved).
*   **Strategy:** Because `us-central1` is hard-capped at 400 CPUs, our 100-node Wolfpack using `g2-standard-4` (4 vCPUs per node = 400 CPUs) **perfectly maxes out** the regional quota limit safely. No room for extra nodes in this region.

## 4. IMMEDIATE NEXT STEPS
1.  **Phase 2 (Cloud Blind Test):** Launch Google Vizier with 100 L4 nodes to search the `[days, ticks_per_day]` grid on GCP.
2.  **Phase 3 (Forge):** Once the optimal constants (lowest FVU) are found, lock them in for an 8x A100 full-training run on the 1.8 billion rows of data (`omega_2d_oracle.pth`).
3.  **Phase 4 (Crucible):** Use `omega_parallel_crucible.py` on Mac Studio for Embarrassingly Parallel Event Study, seeking Asymmetry Payoff > 3.0.

## 5. URGENT DEVIATION: The Data Rebuild
* We are currently blocked on Phase 2 HPO because the 188GB data on GCP lacks  mapping.
* We have executed the Anonymous Math Proof on GCP (Asymmetry Ratio: 1.33) confirming the mathematical viability of the signal.
* We are now executing a 10+ hour local CPU rebuild to regenerate the 188GB matrix WITH symbols. See  for details.

## 5. URGENT DEVIATION: The Data Rebuild
* We are currently blocked on Phase 2 HPO because the 188GB data on GCP lacks `symbol` mapping.
* We have executed the Anonymous Math Proof on GCP (Asymmetry Ratio: 1.33) confirming the mathematical viability of the signal.
* We are now executing a 10+ hour local CPU rebuild to regenerate the 188GB matrix WITH symbols. See `PIPELINE_REBUILD_PLAN.md` for details.

## 6. LINUX1-LX INCIDENT REPORT
* The linux node has experienced catastrophic hardware-level isolation due to repeated AMD UMA memory page faults.
* Read `LINUX1_POST_MORTEM.md` for a complete breakdown of why GPU acceleration is currently impossible without OS-level intervention.
