"""
THE OMEGA PROTOCOL: PURE MATH ANONYMOUS PROOF (GCP CLOUD VERSION)
Paradigm: Epiplexity Validation on Faceless Tensors
Discipline: NO ML. PURE PHYSICS. A-SHARE LIMIT FILTERS.
"""
import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import gcsfs
import time

def verify_math_core_on_shard(file_path, z_score_trigger=-3.0, hold_ticks=2400):
    try:
        df = pd.read_parquet(f"gs://{file_path}", columns=['price', 'order_flow', 'epiplexity'])
    except Exception as e: 
        print(f"Error reading {file_path}: {e}")
        return []

    prices = df['price'].values
    order_flows = df['order_flow'].values
    epi = df['epiplexity'].values 
    
    del df
    
    epi_series = pd.Series(epi)
    rolling_mean = epi_series.rolling(window=4800, min_periods=1000).mean()
    rolling_std = epi_series.rolling(window=4800, min_periods=1000).std()
    
    valid_mask = rolling_std > 1e-8
    z_scores = np.zeros_like(epi_series, dtype=float)
    z_scores[valid_mask] = (epi_series[valid_mask] - rolling_mean[valid_mask]) / rolling_std[valid_mask]
    
    anomalies = np.where(z_scores < z_score_trigger)[0]
    
    trade_log, cooldown_tick, FRICTION = [], 0, 0.0015

    for idx in anomalies:
        if idx < cooldown_tick: continue
        start_idx = max(0, idx - 64)
        if np.sum(order_flows[start_idx : idx + 1]) > 0:
            entry_price = prices[idx]
            exit_idx = min(idx + hold_ticks, len(prices) - 1)
            exit_price = prices[exit_idx]
            
            if entry_price < 1e-5: continue
            pnl = (exit_price - entry_price) / entry_price - (FRICTION * 2)
            if abs(pnl) > 0.21: continue
            trade_log.append(pnl)
            cooldown_tick = exit_idx

    return trade_log

def run_anonymous_proof(gcs_dir):
    fs = gcsfs.GCSFileSystem()
    files = fs.glob(f"{gcs_dir}/*.parquet")
    print(f"🚀 [OMEGA PROOF] Igniting Math Core Verification on {len(files)} Anonymous Cloud Shards...")

    all_pnls = []
    # DRASTICALLY reduce workers to prevent OOM on 416GB machine
    # 60 workers loading 250MB parquet (expanding to ~3GB in memory) = 180GB just for base DF
    # Plus pandas rolling operations which duplicate memory = another 180GB. 
    # That hits the 416GB limit exactly. Let's use 16 workers.
    max_workers = 16
    print(f"⚙️ Using {max_workers} memory-safe cloud workers...")
    
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i, future in enumerate(as_completed([executor.submit(verify_math_core_on_shard, f) for f in files])):
            result = future.result()
            if result:
                all_pnls.extend(result)
            if (i+1) % 50 == 0 or (i+1) == len(files): 
                print(f"   Scanned {i+1}/{len(files)} shards in {time.time()-t0:.1f}s...", flush=True)

    if not all_pnls: 
        print("❌ [VERDICT] No singularities found.", flush=True)
        return

    all_pnls = np.array(all_pnls)
    wins = all_pnls[all_pnls > 0]
    losses = all_pnls[all_pnls <= 0]
    
    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 1e-9
    
    print("\n" + "="*60, flush=True)
    print("🩸 THE MATHEMATICAL VERDICT (ANONYMOUS CLOUD CRUCIBLE)", flush=True)
    print(f"Total Triggers    : {len(all_pnls):,}", flush=True)
    print(f"Win Rate          : {len(wins)/len(all_pnls)*100:.2f}%", flush=True)
    print(f"Average Win       : +{avg_win*100:.2f}%", flush=True)
    print(f"Average Loss      : -{avg_loss*100:.2f}%", flush=True)
    print(f"\n[ASYMMETRY RATIO] : {avg_win / avg_loss:.2f} (Target > 2.0)", flush=True)
    print("="*60, flush=True)

if __name__ == "__main__":
    run_anonymous_proof("omega-pure-data/base_matrix_shards")
