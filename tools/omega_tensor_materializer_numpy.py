import os
import glob
import numpy as np
import polars as pl
from pathlib import Path
import argparse
import time
import gc

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
    print(f"🚀 [FORGE CORE] Firing on pure Numpy (SINGLE THREADED CPU Backend)")

    for filename in parquet_files:
        basename = os.path.basename(filename)
        out_path = os.path.join(output_dir, basename)
        
        if os.path.exists(out_path):
            print(f"⏩ [SKIP] 已经坍缩，跳过: {basename}")
            continue

        try:
            print(f"⏳ [FORGING] 正在提取物理张量: {basename} ...", flush=True)
            t0 = time.time()
            df = pl.scan_parquet(filename).collect()
            print(f"  [DEBUG] Polars collect done in {time.time() - t0:.2f}s, rows: {len(df)}", flush=True)

            t1 = time.time()
            # Calculate missing features in polars before grouping
            df = df.sort(["symbol", "time" if "time" in df.columns else "time_start"])
            df = df.with_columns([
                pl.col("price").diff().over("symbol").fill_null(0.0).cast(pl.Float32).alias("price_change")
            ])
            df = df.with_columns([
                (pl.col("price_change").sign() * pl.col("vol_tick")).cast(pl.Float32).alias("order_flow")
            ])
            
            symbol_groups = df.partition_by("symbol", as_dict=True)
            print(f"  [DEBUG] Polars partition done in {time.time() - t1:.2f}s, groups: {len(symbol_groups)}", flush=True)
            
            del df
            gc.collect()
            
            t2 = time.time()
            all_syms, all_times, all_prices, all_flows, all_price_changes, all_residuals, all_epiplexity = [], [], [], [], [], [], []
            
            for sym, group_df in symbol_groups.items():
                if len(group_df) < 64: continue 
                
                arr_time = group_df["time_start"].to_numpy() if "time_start" in group_df.columns else group_df["time"].to_numpy()
                arr_price = group_df["price"].to_numpy().astype(np.float32)
                arr_order_flow = group_df["order_flow"].to_numpy().astype(np.float32)
                arr_price_change = group_df["price_change"].to_numpy().astype(np.float32)
                arr_market_volume = group_df["vol_tick"].to_numpy().astype(np.float32)
                arr_volatility = group_df["volatility"].to_numpy().astype(np.float32)
                
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

            del symbol_groups
            gc.collect()

            print(f"  [DEBUG] Numpy Forge single-thread compute done in {time.time() - t2:.2f}s", flush=True)
            
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
