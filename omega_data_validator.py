"""
THE OMEGA PROTOCOL: DATA HEALTH VALIDATOR
Execution Target: AMD ROCm (Linux1-lx)
Purpose: Statistically verify the Epiplexity & Compression theory on generated shards.
"""

import os
import pandas as pd
import numpy as np
import sys

def validate_shard(file_path):
    print(f"============================================================")
    print(f"👁️ THE TALEBIAN LENS: {os.path.basename(file_path)}")
    print(f"============================================================")
    
    df = pd.read_parquet(file_path)
    
    print(f"Total Ticks: {len(df):,}")
    print("\n--- SCHEMA & NULL CHECK ---")
    print(df.isna().sum())
    
    # Check Epiplexity Distribution
    epi = df['epiplexity'].replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(epi) == 0:
        print("❌ CRITICAL FAILURE: Epiplexity column is entirely empty or invalid.")
        return
        
    print("\n--- EPIPLEXITY (ENTROPY) DISTRIBUTION ---")
    print(f"Mean: {epi.mean():.6f}")
    print(f"Std:  {epi.std():.6f}")
    print(f"Min:  {epi.min():.6f}  <- [THE SINGULARITY]")
    print(f"25%:  {epi.quantile(0.25):.6f}")
    print(f"50%:  {epi.quantile(0.50):.6f}")
    print(f"99%:  {epi.quantile(0.99):.6f}")
    print(f"Max:  {epi.max():.6f}  <- [RETAIL NOISE / HIGH ENTROPY]")
    
    # The Kolmogorov Compression Theory states that market makers (algorithms)
    # leave a deterministic footprint, causing local entropy to collapse (Epiplexity drops).
    
    # We must filter out market halts (where order flow and price change are 0)
    # because they naturally have zero entropy but are not tradable anomalies.
    active_df = df[df['order_flow'] != 0]
    
    threshold = active_df['epiplexity'].quantile(0.001) # Bottom 0.1% of entropy during active trading
    anomalies = active_df[active_df['epiplexity'] <= threshold].sort_values('epiplexity')
    
    print(f"\n--- DETECTED KOLMOGOROV ANOMALIES (Active Trading Only) ---")
    print(f"Number of Singularity Ticks found: {len(anomalies)}")
    
    if len(anomalies) > 0:
        sample = anomalies.head(5)
        print("\n[Deep Dive: Top 5 Highest Compression Points]")
        for idx, row in sample.iterrows():
            print(f"Epiplexity: {row['epiplexity']:.5f} | Price: {row['price']:.2f} | Flow: {row['order_flow']:.0f} | SRL_Resid: {row['srl_residual']:.5f}")
            
    print("============================================================\n")

if __name__ == "__main__":
    shard_path = sys.argv[1] if len(sys.argv) > 1 else "C:\\omega_pure\\base_matrix_shards_2024\\20240102_fbd5c8b.parquet"
    if os.path.exists(shard_path):
        validate_shard(shard_path)
    else:
        print(f"Waiting for file... {shard_path}")