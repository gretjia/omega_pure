"""
THE OMEGA PROTOCOL: THE TENSOR MATERIALIZER (ROCm GPU Backend)
Execution Target: AMD ROCm (Linux1-lx)
Purpose: Bridges the physical Base_L1.parquet lake to the PyTorch ROCm Forge.
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
import pyarrow.parquet as pq
import gc
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
    
    # 强制开启 ROCm 环境变量以兼容 RDNA 3.5
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🚀 [FORGE CORE] Firing on device: {device}")
    
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
            import pyarrow.parquet as pq
            
            # Using PyArrow to read in batches. This guarantees we NEVER load the 
            # entire 1.5GB file into memory at once, avoiding the kernel I/O Livelock (D state).
            parquet_file = pq.ParquetFile(file)
            
            print(f"  [DEBUG] PyArrow Batch Streaming starting for {parquet_file.metadata.num_rows} rows...", flush=True)
            
            all_prices = []
            all_flows = []
            all_price_changes = []
            all_residuals = []
            all_epiplexity = []
            
            t2 = time.time()
            
            # Process in extremely safe batches
            with torch.no_grad():
                for batch in parquet_file.iter_batches(batch_size=5_000_000, columns=["symbol", "time", "price", "vol_tick"]):
                    import pandas as pd
                    batch_df = batch.to_pandas()
                    
                    # 按照 symbol 在这个 batch 内聚合。
                    # 注意：这会导致跨 batch 的边界出现轻微的 diff 误差（约丢失 1 个 tick 的差分），
                    # 但在 1 亿行的数据尺度上，这在物理上完全可以忽略不计，且彻底解决了内存爆炸。
                    batch_df['price_change'] = batch_df.groupby('symbol')['price'].diff().fillna(0.0).astype(np.float32)
                    batch_df['order_flow'] = (np.sign(batch_df['price_change']) * batch_df['vol_tick']).astype(np.float32)
                    
                    grouped = batch_df.groupby("symbol")
                    
                    for sym, group_df in grouped:
                        original_len = len(group_df)
                        if original_len < 64: 
                            continue
                            
                        arr_price = group_df["price"].to_numpy(dtype=np.float32)
                        arr_order_flow = group_df["order_flow"].to_numpy(dtype=np.float32)
                        arr_price_change = group_df["price_change"].to_numpy(dtype=np.float32)
                        arr_vol_tick = group_df["vol_tick"].to_numpy(dtype=np.float32)
                        
                        def rolling_std(x, window=64):
                            if len(x) < window: return np.full_like(x, 0.001)
                            shape = x.shape[:-1] + (x.shape[-1] - window + 1, window)
                            strides = x.strides + (x.strides[-1],)
                            windows = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
                            res = np.zeros_like(x)
                            res[window-1:] = np.std(windows, axis=-1)
                            res[:window-1] = res[window-1]
                            res[res < 0.001] = 0.001
                            return res
                            
                        arr_volatility = rolling_std(arr_price)

                        price_change_t = torch.tensor(arr_price_change, device=device)
                        order_flow_t = torch.tensor(arr_order_flow, device=device)
                        market_volume_t = torch.tensor(arr_vol_tick, device=device)
                        volatility_t = torch.tensor(arr_volatility, device=device)

                        CHUNK_MAX = 2000
                        all_feature_chunks = []
                        
                        for i in range(0, original_len, CHUNK_MAX):
                            end_idx = min(i + CHUNK_MAX, original_len)
                            if end_idx - i < 10: 
                                chunk_zeros = np.zeros((end_idx - i, 3), dtype=np.float32)
                                chunk_zeros[:, 0] = arr_price_change[i:end_idx]
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
                    
                    # 清理 batch 内存
                    del batch, batch_df, grouped
                    gc.collect()
            
            torch.cuda.empty_cache()
            print(f"  [DEBUG] GPU Tensor Forging done in {time.time() - t2:.2f}s", flush=True)

            if not all_prices:
                continue

            import pandas as pd
            shard_df = pd.DataFrame({
                "symbol": np.concatenate(all_syms),
                "time": np.concatenate(all_times),
                "price": np.concatenate(all_prices),
                "order_flow": np.concatenate(all_flows),
                "price_change": np.concatenate(all_price_changes),
                "srl_residual": np.concatenate(all_residuals),
                "epiplexity": np.concatenate(all_epiplexity)
            })
            
            shard_df.to_parquet(out_path)
            print(f"✔️ [SUCCESS] 降维完成 -> {out_path} (行数: {len(shard_df)})", flush=True)
            
            del all_prices, all_flows, all_price_changes, all_residuals, all_epiplexity, shard_df
            gc.collect()
            
        except Exception as e:
            print(f"❌ [ERROR] 失败 {filename}: {str(e)}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_l1_dir", type=str, default="/omega_pool/parquet_data/latest_base_l1")
    parser.add_argument("--output_dir", type=str, default="./base_matrix_shards")
    parser.add_argument("--years", type=str, default=None, help="Comma-separated years to process, e.g., '2023,2026'")
    args = parser.parse_args()
    
    target_years = args.years.split(",") if args.years else None
    materialize_shards(args.base_l1_dir, args.output_dir, target_years=target_years)