"""
THE OMEGA PROTOCOL: THE TENSOR MATERIALIZER (DirectML GPU Backend)
Execution Target: AMD AI Max 395 (Windows1-w1)
Purpose: Bridges the physical Base_L1.parquet lake to the PyTorch DirectML Forge.
Discipline: STRICT VECTORIZATION. NO OOP. EXTREME GPU ACCELERATION.
"""

import os
import glob
import numpy as np
import polars as pl
from pathlib import Path
import argparse
import time
import torch
import torch_directml
import warnings
warnings.filterwarnings("ignore", "overflow encountered in exp")

from omega_epiplexity_forge_pytorch import forge_epiplexity_tensor

def materialize_shards(base_l1_dir: str, output_dir: str, target_years: list = None):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    parquet_files = sorted(glob.glob(f"{base_l1_dir}/**/*.parquet", recursive=True))
    if not parquet_files:
        print(f"[FATAL] 没找到 Base_L1 数据：{base_l1_dir}")
        return
        
    print(f"[MATERIALIZER] 发现 {len(parquet_files)} 个 Base_L1 碎片。准备坍缩...")
    
    device = torch_directml.device()
    print(f"🚀 [FORGE CORE] Firing on DirectML device: {device} (Radeon 8060S)")
    
    for file in parquet_files:
        filename = os.path.basename(file)
        
        if target_years:
            if not any(filename.startswith(str(year)) for year in target_years):
                continue

        out_path = os.path.join(output_dir, filename)
        
        if os.path.exists(out_path):
            print(f"⏩ [SKIP] 已经坍缩，跳过: {filename}")
            continue
            
        print(f"⏳ [FORGING] 正在提取物理张量: {filename} ...", flush=True)
        t0 = time.time()
        
        try:
            df = pl.scan_parquet(file).select(
                ["symbol", "time", "price", "vol_tick"]
            ).with_columns([
                pl.col("price").diff().fill_null(0.0).over("symbol").alias("price_change"),
                (pl.col("price").diff().sign() * pl.col("vol_tick")).fill_null(0.0).over("symbol").alias("order_flow"),
                pl.col("price").rolling_std(window_size=64).fill_null(0.001).over("symbol").alias("volatility")
            ]).collect()
            
            print(f"  [DEBUG] Polars collect done in {time.time() - t0:.2f}s, rows: {len(df)}", flush=True)
            if len(df) < 128:
                continue

            t1 = time.time()
            symbol_groups = df.partition_by("symbol", as_dict=True)
            print(f"  [DEBUG] Polars partition done in {time.time() - t1:.2f}s, groups: {len(symbol_groups)}", flush=True)
            
            all_prices = []
            all_flows = []
            all_price_changes = []
            all_residuals = []
            all_epiplexity = []
            
            t2 = time.time()
            # PyTorch DirectML Backend
            with torch.no_grad():
                for idx, (sym, group_df) in enumerate(symbol_groups.items()):
                    if idx > 0 and idx % 1000 == 0:
                        print(f"  [DEBUG] Processed symbol {idx}/{len(symbol_groups)}...", flush=True)
                        # No empty_cache() needed for DML like CUDA usually, but good practice
                        # torch_directml does not have a formal empty_cache, so we rely on GC.
                        
                    original_len = len(group_df)
                    if original_len < 64: 
                        continue
                        
                    # 抓取数据到 CPU
                    arr_price = group_df["price"].to_numpy().astype(np.float32)
                    arr_order_flow = group_df["order_flow"].to_numpy().astype(np.float32)
                    
                    # 生成 Tensor 并直接发往 DirectML GPU
                    price_change_t = torch.tensor(group_df["price_change"].to_numpy().astype(np.float32), device=device)
                    order_flow_t = torch.tensor(arr_order_flow, device=device)
                    market_volume_t = torch.tensor(group_df["vol_tick"].to_numpy().astype(np.float32), device=device)
                    volatility_t = torch.tensor(group_df["volatility"].to_numpy().astype(np.float32), device=device)

                    # Chunking 以防 Windows DML OOM
                    CHUNK_MAX = 2000
                    all_feature_chunks = []
                    
                    for i in range(0, original_len, CHUNK_MAX):
                        end_idx = min(i + CHUNK_MAX, original_len)
                        if end_idx - i < 10: 
                            chunk_zeros = np.zeros((end_idx - i, 3), dtype=np.float32)
                            chunk_zeros[:, 0] = arr_price[i:end_idx]
                            all_feature_chunks.append(chunk_zeros)
                            continue
                            
                        pc_chunk = price_change_t[i:end_idx]
                        of_chunk = order_flow_t[i:end_idx]
                        mv_chunk = market_volume_t[i:end_idx]
                        vol_chunk = volatility_t[i:end_idx]
                        
                        feature_tensor_chunk = forge_epiplexity_tensor(
                            price_change=pc_chunk,
                            order_flow=of_chunk,
                            market_volume=mv_chunk,
                            volatility=vol_chunk,
                            dim=10, delay=1
                        )
                        all_feature_chunks.append(feature_tensor_chunk.cpu().numpy())
                        del pc_chunk, of_chunk, mv_chunk, vol_chunk, feature_tensor_chunk
                    
                    del price_change_t, order_flow_t, market_volume_t, volatility_t
                    
                    feature_np = np.concatenate(all_feature_chunks, axis=0)
                    
                    all_prices.append(arr_price)
                    all_flows.append(arr_order_flow)
                    all_price_changes.append(feature_np[:, 0])
                    all_residuals.append(feature_np[:, 1])
                    all_epiplexity.append(feature_np[:, 2])
            
            print(f"  [DEBUG] GPU Tensor Forging done in {time.time() - t2:.2f}s", flush=True)

            if not all_prices:
                continue

            import pandas as pd
            shard_df = pd.DataFrame({
                "price": np.concatenate(all_prices),
                "order_flow": np.concatenate(all_flows),
                "price_change": np.concatenate(all_price_changes),
                "srl_residual": np.concatenate(all_residuals),
                "epiplexity": np.concatenate(all_epiplexity)
            })
            
            shard_df.to_parquet(out_path)
            print(f"✔️ [SUCCESS] 降维完成 -> {out_path} (行数: {len(shard_df)})", flush=True)
            
        except Exception as e:
            print(f"❌ [ERROR] 失败 {filename}: {str(e)}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_l1_dir", type=str, default="D:\\Omega_frames\\latest_base_l1")
    parser.add_argument("--output_dir", type=str, default="C:\\omega_pure\\base_matrix_shards_2024")
    parser.add_argument("--years", type=str, default=None, help="Comma-separated years to process, e.g., '2024,2025'")
    args = parser.parse_args()
    
    target_years = args.years.split(",") if args.years else None
    materialize_shards(args.base_l1_dir, args.output_dir, target_years=target_years)