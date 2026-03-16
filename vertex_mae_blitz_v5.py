"""
THE OMEGA PROTOCOL: THE KOLMOGOROV COMPRESSOR (V5 TRUE PHYSICS EDITION)
Paradigm: Macro-Band Accumulation Discovery (1 to 3 Weeks)
Discipline: STRICT TICKER BOUNDARIES. TRUE TIME SCALING. NO ALIASING.
"""
import sys
import argparse
import gc
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import gcsfs
import hypertune  
from torch.utils.data import Dataset, DataLoader
from omega_2d_folded_mae import SpatioTemporal2DMAE, TimeFoldedDataset

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def load_ticker_data_true_physics(gcs_path, limit_files=100):
    print(f"[BLITZ-V5] Loading {limit_files} Ticker Shards for true physical timeline...", flush=True)
    fs = gcsfs.GCSFileSystem()
    files = sorted(fs.glob(f"{gcs_path}/*.parquet"))
    if not files: raise ValueError(f"FATAL: No shards at {gcs_path}")
    
    # 🚨 致命安检：拦截按“天”打包的错误数据 (基于典型日期命名格式如 20250102_xx.parquet)
    if any("202" in f.split('/')[-1][:4] and len(f.split('/')[-1]) > 15 for f in files[:3]):
        print(f"🚨 [FATAL ERROR] Detected Daily Shards (e.g. 20250102_xxx.parquet).", flush=True)
        print(f"You MUST run tools/repack_to_ticker_shards.py first to create per-stock files!", flush=True)
        sys.exit(1)
        
    files = files[:limit_files]
    shard_info = []
    total_rows = 0
    columns_to_load = ['price_change', 'srl_residual', 'epiplexity']
    
    import pyarrow.parquet as pq
    for f in files:
        with fs.open(f) as pf:
            rows = pq.read_metadata(pf).num_rows
            shard_info.append((f, rows))
            total_rows += rows
            
    print(f"[BLITZ-V5] Total valid ticker rows: {total_rows:,}. Allocating RAM...", flush=True)
    full_data = torch.empty((total_rows, 3), dtype=torch.float32)
    
    # 记录每只股票的物理边界
    train_boundaries = []
    val_boundaries = []
    current_pos = 0
    sum_val = np.zeros(3, dtype=np.float64)
    sum_sq = np.zeros(3, dtype=np.float64)
    train_rows_total = 0
    
    for f, rows in shard_info:
        df = pd.read_parquet(f"gs://{f}", columns=columns_to_load, engine='pyarrow')
        arr = df.values.astype(np.float64)
        
        # 80/20 时间前向切分 (每只股票各自取前 80% 用于训练)
        split_idx = int(rows * 0.8)
        if split_idx > 0:
            sum_val += arr[:split_idx].sum(axis=0)
            sum_sq += (arr[:split_idx]**2).sum(axis=0)
            train_rows_total += split_idx
            
        shard_tensor = torch.from_numpy(df.values).float()
        full_data[current_pos : current_pos + rows] = shard_tensor
        
        train_boundaries.append((current_pos, current_pos + split_idx))
        val_boundaries.append((current_pos + split_idx, current_pos + rows))
        
        current_pos += rows
        del df, arr, shard_tensor; gc.collect()
        
    mean = sum_val / max(1, train_rows_total)
    std = np.sqrt(np.maximum((sum_sq / max(1, train_rows_total)) - mean**2, 1e-12)) + 1e-8
    
    print(f"[BLITZ-V5] Normalizing In-Place...", flush=True)
    mean_t = torch.from_numpy(mean).float()
    std_t = torch.from_numpy(std).float()
    full_data.sub_(mean_t).div_(std_t)
    
    full_data.share_memory_()
    return full_data, train_boundaries, val_boundaries



@torch.no_grad()
def fast_validate(model, val_iter, device, max_batches=50):
    model.eval()
    val_fvu_total, count = 0.0, 0
    for _ in range(max_batches):
        try: batch = next(val_iter)
        except StopIteration: break
        batch = batch.to(device, non_blocking=True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            _, batch_fvu = model(batch)
        val_fvu_total += batch_fvu.item()
        count += 1
    model.train()
    return val_fvu_total / max(1, count)

def forge_compressor(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hpt_client = hypertune.HyperTune()
    accum_steps = max(1, args.logical_batch_size // args.micro_batch_size)
    
    print(f"[BLITZ-V5] Physics Engaged | Days: {args.days} | Ticks/Day: {args.ticks_per_day}", flush=True)
    full_data, train_bounds, val_bounds = load_ticker_data_true_physics(args.gcs_input, limit_files=100)
    
    train_loader = DataLoader(TimeFoldedDataset(full_data, train_bounds, args.days, args.ticks_per_day), batch_size=args.micro_batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(TimeFoldedDataset(full_data, val_bounds, args.days, args.ticks_per_day), batch_size=args.micro_batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    model = SpatioTemporal2DMAE(days=args.days, ticks_per_day=args.ticks_per_day).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    def infinite_loader(dl):
        while True:
            for b in dl: yield b
            
    train_iter = infinite_loader(train_loader)
    val_iter = infinite_loader(val_loader)
    model.train()
    
    for logical_step in range(1, args.max_steps + 1):
        optimizer.zero_grad()
        for _ in range(accum_steps):
            try: batch = next(train_iter).to(device, non_blocking=True)
            except StopIteration: break
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                _, fvu_loss = model(batch)
                loss = fvu_loss / accum_steps
            loss.backward()
            
            if torch.isnan(fvu_loss) or torch.isinf(fvu_loss):
                print(f"🚨 [FATAL] NaN detected. Invalid Physics Days/Ticks. Suiciding.", flush=True)
                hpt_client.report_hyperparameter_tuning_metric(hyperparameter_metric_tag='val_fvu', metric_value=9999.0, global_step=logical_step)
                sys.exit(0)
                
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if logical_step % args.report_freq == 0:
            val_fvu = fast_validate(model, val_iter, device, max_batches=50)
            print(f"⚡ [SONAR] Step {logical_step:04d}/{args.max_steps} | Val FVU: {val_fvu:.5f}", flush=True)
            sys.stdout.flush()
            hpt_client.report_hyperparameter_tuning_metric(hyperparameter_metric_tag='val_fvu', metric_value=val_fvu, global_step=logical_step)
            
            # 放宽判决线：15天宏观预测极难，1.05 以内都算有潜力
            if logical_step >= 1000 and val_fvu >= 1.05:
                print(f"☠️ [EXECUTION] Weak config. Pulling the plug.", flush=True)
                hpt_client.report_hyperparameter_tuning_metric(hyperparameter_metric_tag='val_fvu', metric_value=9999.0, global_step=args.max_steps)
                sys.exit(0)
                
    print(f"✅ Trial Complete. Maximum steps reached.", flush=True)

if __name__ == "__main__":
    try: torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError: pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--gcs_input", type=str, required=True)
    parser.add_argument("--logical_batch_size", type=int, default=4096)
    parser.add_argument("--micro_batch_size", type=int, default=64) # 应对 512 的显存消耗
    parser.add_argument("--max_steps", type=int, default=2000) # 提速 HPO
    parser.add_argument("--report_freq", type=int, default=500)
    parser.add_argument("--days", type=int, default=15)
    parser.add_argument("--ticks_per_day", type=int, default=64)
    args, _ = parser.parse_known_args()
    forge_compressor(args)
