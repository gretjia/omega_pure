"""
THE OMEGA PROTOCOL: PURE MATH ANONYMOUS PROOF (NUMPY VECTORIZED)
Paradigm: Epiplexity Validation on Faceless Tensors
Discipline: PURE NUMPY. STRIDED ROLLING WINDOWS. OOM IMMUNITY.
"""
import os
import numpy as np
import pandas as pd
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

def strided_rolling_std(a, window):
    down_a = a[::10]
    w = window // 10
    if len(down_a) < w:
        return np.ones_like(a) * np.nan
    shape = down_a.shape[:-1] + (down_a.shape[-1] - w + 1, w)
    strides = down_a.strides + (down_a.strides[-1],)
    strided = np.lib.stride_tricks.as_strided(down_a, shape=shape, strides=strides)
    roll_mean = np.nanmean(strided, axis=-1)
    roll_std = np.nanstd(strided, axis=-1)
    pad_mean = np.pad(roll_mean, (w - 1, 0), mode='edge')
    pad_std = np.pad(roll_std, (w - 1, 0), mode='edge')
    full_mean = np.repeat(pad_mean, 10)[:len(a)]
    full_std = np.repeat(pad_std, 10)[:len(a)]
    if len(full_mean) < len(a):
        full_mean = np.pad(full_mean, (0, len(a) - len(full_mean)), mode='edge')
        full_std = np.pad(full_std, (0, len(a) - len(full_std)), mode='edge')
    return full_mean, full_std

def verify_math_core_on_shard(file_path, z_score_trigger=-3.0, hold_ticks=2400):
    try:
        df = pd.read_parquet(file_path, columns=['price', 'order_flow', 'epiplexity'])
    except Exception: return []

    prices = df['price'].values
    order_flows = df['order_flow'].values
    epi = df['epiplexity'].values 
    del df
    
    roll_mean, roll_std = strided_rolling_std(epi, window=4800)
    valid_mask = roll_std > 1e-8
    z_scores = np.zeros_like(epi, dtype=float)
    z_scores[valid_mask] = (epi[valid_mask] - roll_mean[valid_mask]) / roll_std[valid_mask]
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

def run_anonymous_proof(data_dir):
    files = sorted(glob(f"{data_dir}/*.parquet"))[:50] 
    print(f"🚀 [OMEGA PROOF] Igniting Vectorized Math Core Verification on {len(files)} Anonymous Shards...")

    all_pnls = []
    # 极度压制并发！即使是 Numpy，多开进程也会拷贝内存导致机器崩溃。
    # 只用 2 个进程跑，稳妥第一。
    max_workers = 2
    print(f"⚙️ Using {max_workers} numpy-vectorized workers (CRITICAL SAFETY LIMIT)...")
    
    start_t = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i, future in enumerate(as_completed([executor.submit(verify_math_core_on_shard, f) for f in files])):
            result = future.result()
            if result:
                all_pnls.extend(result)
            if (i+1) % 5 == 0: 
                print(f"   Scanned {i+1}/{len(files)} shards... Time: {time.time()-start_t:.1f}s")

    if not all_pnls: return print("❌ [VERDICT] No singularities found.")

    all_pnls = np.array(all_pnls)
    wins, losses = all_pnls[all_pnls > 0], all_pnls[all_pnls <= 0]
    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 1e-9
    
    print("\n" + "="*60)
    print("🩸 THE MATHEMATICAL VERDICT (ANONYMOUS CRUCIBLE)")
    print(f"Total Triggers    : {len(all_pnls):,}")
    print(f"Average Win       : +{avg_win*100:.2f}%")
    print(f"Average Loss      : -{avg_loss*100:.2f}%")
    print(f"\n[ASYMMETRY RATIO] : {avg_win / avg_loss:.2f} (Target > 2.0)")
    print("="*60)

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./base_matrix_shards"
    run_anonymous_proof(data_dir)
