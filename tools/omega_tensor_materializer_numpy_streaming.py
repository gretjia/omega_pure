import os
import glob
import numpy as np
from pathlib import Path
import argparse
import time
import gc
import pyarrow.parquet as pq

def np_compute_srl_residual(price_change, order_flow, market_volume, volatility, gamma=1.0):
    expected_impact = gamma * volatility * np.sign(order_flow) * np.sqrt(np.abs(order_flow) / (market_volume + 1e-8))
    return price_change - expected_impact

def np_compute_epiplexity(price_change, order_flow, alpha=0.5):
    p_norm = (price_change - np.mean(price_change)) / (np.std(price_change) + 1e-8)
    q_norm = (order_flow - np.mean(order_flow)) / (np.std(order_flow) + 1e-8)
    divergence = np.abs(p_norm - q_norm)
    entropy_penalty = divergence * np.log1p(divergence)
    return np.exp(-alpha * entropy_penalty)

def materialize_shards(base_l1_dir: str, output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    parquet_files = sorted(glob.glob(f"{base_l1_dir}/**/*.parquet", recursive=True))
    if not parquet_files:
        print(f"[FATAL] 没找到 Base_L1 数据：{base_l1_dir}")
        return

    print(f"[MATERIALIZER] 发现 {len(parquet_files)} 个 Base_L1 碎片。准备坍缩...")
    print(f"🚀 [FORGE CORE] Firing on pure Numpy (STREAMING CPU Backend)")

    for filename in parquet_files:
        basename = os.path.basename(filename)
        out_path = os.path.join(output_dir, basename)
        
        if os.path.exists(out_path):
            print(f"⏩ [SKIP] 已经坍缩，跳过: {basename}")
            continue

        try:
            print(f"⏳ [FORGING] 正在提取物理张量: {basename} ...", flush=True)
            t0 = time.time()
            
            parquet_file = pq.ParquetFile(filename)
            total_rows = parquet_file.metadata.num_rows
            
            all_syms, all_times, all_prices, all_flows, all_price_changes, all_residuals, all_epiplexity = [], [], [], [], [], [], []
            
            # Streaming read to prevent OOM
            rows_processed = 0
            for batch in parquet_file.iter_batches(batch_size=1_000_000):
                df = batch.to_pandas()
                
                # Calculate required columns locally for the batch
                df = df.sort_values(by=["symbol", "time" if "time" in df.columns else "time_start"])
                df['price_change'] = df.groupby('symbol')['price'].diff().fillna(0.0).astype(np.float32)
                df['order_flow'] = (np.sign(df['price_change']) * df['vol_tick']).astype(np.float32)
                
                grouped = df.groupby("symbol")
                
                for sym, group_df in grouped:
                    if len(group_df) < 64: continue 
                    
                    arr_time = group_df["time_start"].to_numpy() if "time_start" in group_df.columns else group_df["time"].to_numpy()
                    arr_price = group_df["price"].to_numpy().astype(np.float32)
                    arr_order_flow = group_df["order_flow"].to_numpy().astype(np.float32)
                    arr_price_change = group_df["price_change"].to_numpy().astype(np.float32)
                    arr_market_volume = group_df["vol_tick"].to_numpy().astype(np.float32)
                    
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
                    
                    MAX_TICKS = 5500
                    original_len = len(arr_price)
                    pad_size = MAX_TICKS - original_len
                    
                    if pad_size > 0:
                        arr_time_pad = np.pad(arr_time, (0, pad_size), mode='constant', constant_values=np.datetime64('NaT')) if np.issubdtype(arr_time.dtype, np.datetime64) else np.pad(arr_time, (0, pad_size), mode='constant', constant_values='')
                        arr_price_change_pad = np.pad(arr_price_change, (0, pad_size), mode='constant')
                        arr_order_flow_pad = np.pad(arr_order_flow, (0, pad_size), mode='constant')
                        arr_market_volume_pad = np.pad(arr_market_volume, (0, pad_size), mode='constant')
                        arr_volatility_pad = np.pad(arr_volatility, (0, pad_size), mode='constant', constant_values=1.0)
                    else:
                        arr_time_pad = arr_time[:MAX_TICKS]
                        arr_price_change_pad = arr_price_change[:MAX_TICKS]
                        arr_order_flow_pad = arr_order_flow[:MAX_TICKS]
                        arr_market_volume_pad = arr_market_volume[:MAX_TICKS]
                        arr_volatility_pad = arr_volatility[:MAX_TICKS]

                    srl_resid = np_compute_srl_residual(arr_price_change_pad, arr_order_flow_pad, arr_market_volume_pad, arr_volatility_pad)
                    epi = np_compute_epiplexity(arr_price_change_pad, arr_order_flow_pad)
                    arr_sym = np.full(original_len, sym, dtype=object)

                    all_syms.append(arr_sym)
                    all_times.append(arr_time_pad[:original_len] if pad_size > 0 else arr_time[:original_len])
                    all_prices.append(arr_price[:original_len])
                    all_flows.append(arr_order_flow[:original_len])
                    all_price_changes.append(arr_price_change_pad[:original_len])
                    all_residuals.append(srl_resid[:original_len])
                    all_epiplexity.append(epi[:original_len])

                # Aggressive GC
                del df, grouped
                gc.collect()
                
                rows_processed += len(batch)
                print(f"    - Processed batch: {rows_processed}/{total_rows} rows", flush=True)

            print(f"  [DEBUG] Streaming Numpy Forge done in {time.time() - t0:.2f}s", flush=True)
            
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
            
            del all_syms, all_times, all_prices, all_flows, all_price_changes, all_residuals, all_epiplexity, shard_df
            gc.collect()
            
        except Exception as e:
            print(f"❌ [ERROR] 失败 {filename}: {str(e)}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_l1_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    materialize_shards(args.base_l1_dir, args.output_dir)