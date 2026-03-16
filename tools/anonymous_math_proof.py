"""
THE OMEGA PROTOCOL: PURE MATH ANONYMOUS PROOF
Paradigm: Epiplexity Validation on Faceless Tensors
Discipline: NO ML. PURE PHYSICS. A-SHARE LIMIT FILTERS.
"""
import os
import numpy as np
import pandas as pd
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

def verify_math_core_on_shard(file_path, z_score_trigger=-3.0, hold_ticks=2400):
    try:
        # 极速加载需要的数学真理列
        df = pd.read_parquet(file_path, columns=['price', 'order_flow', 'srl_residual', 'epiplexity'])
    except Exception: return []

    prices, order_flows, epi = df['price'].values, df['order_flow'].values, df['epiplexity'].values 
    
    # 寻找低于均值 3 个标准差的极低熵瞬间 (主力算法单介入的特征)
    epi_series = pd.Series(epi)
    rolling_mean = epi_series.rolling(window=4800, min_periods=1000).mean()
    rolling_std = epi_series.rolling(window=4800, min_periods=1000).std()
    
    z_scores = (epi_series - rolling_mean) / (rolling_std + 1e-8)
    anomalies = np.where(z_scores < z_score_trigger)[0]
    
    trade_log, cooldown_tick, FRICTION = [], 0, 0.0015

    for idx in anomalies:
        if idx < cooldown_tick: continue # 冷却期防连发
            
        # 验证物理发力方向：只有主力确定性净流入的低熵时刻才做多
        start_idx = max(0, idx - 64)
        if np.sum(order_flows[start_idx : idx + 1]) > 0:
            entry_price = prices[idx]
            exit_idx = min(idx + hold_ticks, len(prices) - 1)
            exit_price = prices[exit_idx]
            
            if entry_price < 1e-5: continue
            pnl = (exit_price - entry_price) / entry_price - (FRICTION * 2)
            
            # 🚀 [架构师防污染绝杀]：半天收益率超 21%，绝对是跨越了股票拼接边界！剔除！
            if abs(pnl) > 0.21: continue
            trade_log.append(pnl)
            cooldown_tick = exit_idx

    return trade_log

def run_anonymous_proof(data_dir):
    files = sorted(glob(f"{data_dir}/*.parquet"))[:5] # 极度缩小测试范围，防止再次引爆内存
    print(f"🚀 [OMEGA PROOF] Igniting Math Core Verification on {len(files)} Anonymous Shards...")

    all_pnls = []
    # 强制限制 worker 数量为 2，防止内存爆炸
    with ProcessPoolExecutor(max_workers=2) as executor:
        for i, future in enumerate(as_completed([executor.submit(verify_math_core_on_shard, f) for f in files])):
            all_pnls.extend(future.result())
            print(f"   Scanned {i+1}/{len(files)} shards...")

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
