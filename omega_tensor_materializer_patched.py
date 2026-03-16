"""
THE OMEGA PROTOCOL: THE TENSOR MATERIALIZER
Execution Target: AMD ROCm (Linux1-lx / Windows1-w1)
Purpose: Bridges the physical Base_L1.parquet lake to the JAX Epiplexity Forge.
Discipline: STRICT VECTORIZATION. NO OOP. NO PHYSICS MATH.
"""

import os
import glob
import numpy as np
import polars as pl
from pathlib import Path
import warnings
import argparse
import time
import gc
warnings.filterwarnings("ignore", "overflow encountered in exp")
import multiprocessing as mp

# 导入我们在 Step 2 写好的 Pure Numpy Forge
from omega_epiplexity_forge import forge_epiplexity_tensor

# 定义一个模块级纯函数供多进程映射
def process_single_symbol(data_pack):
    sym, arr_time, arr_price, arr_order_flow, arr_price_change, arr_market_volume, arr_volatility = data_pack
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
        original_len = MAX_TICKS

    feature_tensor_pad = forge_epiplexity_tensor(
        price_change=arr_price_change_pad,
        order_flow=arr_order_flow_pad,
        market_volume=arr_market_volume_pad,
        volatility=arr_volatility_pad,
        dim=10, delay=1
    )
    
    feature_np = np.asarray(feature_tensor_pad)[:original_len]
    arr_sym = np.full(original_len, sym, dtype=object)
    
    return (
        arr_sym,
        arr_time_pad[:original_len] if pad_size > 0 else arr_time[:original_len],
        arr_price[:original_len],
        arr_order_flow[:original_len],
        feature_np[:, 0],
        feature_np[:, 1],
        feature_np[:, 2]
    )

def materialize_shards(base_l1_dir: str, output_dir: str, target_years: list = None):
    """
    流式读取 Base_L1，直接通过 Polars 提取差分张量，然后灌入 Numpy 降维。
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Base_L1 files might be nested under host=linux1/ or host=windows1/
    # so we search recursively
    parquet_files = sorted(glob.glob(f"{base_l1_dir}/**/*.parquet", recursive=True))
    if not parquet_files:
        print(f"[FATAL] 没找到 Base_L1 数据：{base_l1_dir}")
        return
        
    print(f"[MATERIALIZER] 发现 {len(parquet_files)} 个 Base_L1 碎片。准备坍缩...")
    
    # 限制并发以避免吃满 128G 内存。
    # Numpy 已经多线程优化了。进程开得太多反而会造成系统 Swap 崩溃。
    # 最佳实践：16 进程。
    WORKER_COUNT = 16
    
    for file in parquet_files:
        filename = os.path.basename(file)
        
        # 按照年份过滤 (如 20230101_xxx.parquet)
        if target_years:
            if not any(filename.startswith(str(year)) for year in target_years):
                continue

        out_path = os.path.join(output_dir, filename)
        
        if os.path.exists(out_path):
            print(f"⏩ [SKIP] 已经坍缩，跳过: {filename}")
            continue
            
        print(f"⏳ [FORGING] 正在提取物理张量: {filename} ...", flush=True)
        t0 = time.time()
        # 1. 极速 Polars 物理列提取 & 差分运算
        # 我们只读取 Numpy 需要的基础列，不全量加载，防 OOM
        # 关键修正：必须加上 over("symbol") 以防止不同股票串帧，引发物理谬误
        try:
            df = pl.scan_parquet(file).select(
                ["symbol", "time", "price", "vol_tick"]
            ).with_columns([
                # 计算价格变化量
                pl.col("price").diff().fill_null(0.0).over("symbol").alias("price_change"),
                
                # 订单流 = 价格变化方向 * 交易量 (物理冲击力近似)
                (pl.col("price").diff().sign() * pl.col("vol_tick")).fill_null(0.0).over("symbol").alias("order_flow"),
                
                # 64个Tick的滚动波动率
                pl.col("price").rolling_std(window_size=64).fill_null(0.001).over("symbol").alias("volatility")
            ]).collect()
            
            print(f"  [DEBUG] Polars collect done in {time.time() - t0:.2f}s, rows: {len(df)}", flush=True)
            if len(df) < 128:
                continue

            # 2. 必须按 symbol 分组处理，否则 O(N^2) 拓扑距离矩阵会瞬间撑爆 128G 显存！
            t1 = time.time()
            symbol_groups = df.partition_by("symbol", as_dict=True)
            print(f"  [DEBUG] Polars partition done in {time.time() - t1:.2f}s, groups: {len(symbol_groups)}", flush=True)
            
            # 释放主 DataFrame 内存，只留字典
            del df
            gc.collect()
            
            t2 = time.time()
            pack_list = []
            for sym, group_df in symbol_groups.items():
                if len(group_df) < 64: 
                    continue # 忽略交易太不活跃的标的
                
                # 零拷贝转 numpy
                arr_time = group_df["time_start"].to_numpy() if "time_start" in group_df.columns else group_df["time"].to_numpy()
                arr_price = group_df["price"].to_numpy().astype(np.float32)
                arr_order_flow = group_df["order_flow"].to_numpy().astype(np.float32)
                arr_price_change = group_df["price_change"].to_numpy().astype(np.float32)
                arr_market_volume = group_df["vol_tick"].to_numpy().astype(np.float32)
                arr_volatility = group_df["volatility"].to_numpy().astype(np.float32)
                
                pack_list.append((sym, arr_time, arr_price, arr_order_flow, arr_price_change, arr_market_volume, arr_volatility))

            # 释放分组字典内存
            del symbol_groups
            gc.collect()

            print(f"  [DEBUG] Data packed in {time.time() - t2:.2f}s. Submitting to Pool with {WORKER_COUNT} cores...", flush=True)
            
            if not pack_list:
                continue

            t3 = time.time()
            with mp.Pool(processes=WORKER_COUNT) as pool:
                # 使用 chunksize=50 避免频繁 IPC 开销
                results = pool.map(process_single_symbol, pack_list, chunksize=50)
            
            print(f"  [DEBUG] Epiplexity Forge parallel compute done in {time.time() - t3:.2f}s", flush=True)
            
            # Unpack results
            all_syms, all_times, all_prices, all_flows, all_price_changes, all_residuals, all_epiplexity = zip(*results)

            # 3. 封装降维后的拓扑碎片并落盘
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
            
            del pack_list, results, all_prices, all_flows, all_price_changes, all_residuals, all_epiplexity, shard_df
            gc.collect()
            
        except Exception as e:
            print(f"❌ [ERROR] 失败 {filename}: {str(e)}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 默认指向 Linux 的 Base L1 挂载点，如果在 Windows 运行则传入对应的路径
    parser.add_argument("--base_l1_dir", type=str, default="/omega_pool/parquet_data/latest_base_l1")
    parser.add_argument("--output_dir", type=str, default="./base_matrix_shards")
    parser.add_argument("--years", type=str, default=None, help="Comma-separated years to process, e.g., '2023,2026'")
    args = parser.parse_args()
    
    target_years = args.years.split(",") if args.years else None
    materialize_shards(args.base_l1_dir, args.output_dir, target_years=target_years)