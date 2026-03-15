"""
THE OMEGA PROTOCOL: THE KOLMOGOROV COMPRESSOR (V3 STRATIFIED BLITZ - 2025 EDITION)
Paradigm: Scale-Invariant Epiplexity Discovery
Discipline: STRATIFIED SAMPLING. ZERO OOM on 16GB RAM.
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

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class EpiplexityMAE(nn.Module):
    def __init__(self, feature_dim=3, seq_len=64, embed_dim=64, num_heads=4, depth=4, mask_ratio=0.70):
        super().__init__()
        self.seq_len = seq_len
        self.mask_ratio = mask_ratio
        self.input_proj = nn.Linear(feature_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.output_proj = nn.Linear(embed_dim, feature_dim)

    def forward(self, x):
        B, T, F = x.shape
        x_emb = self.input_proj(x) + self.pos_embed
        noise = torch.rand(B, T, device=x.device)
        mask = (noise < self.mask_ratio).bool()
        x_masked = x_emb.clone()
        x_masked[mask] = self.mask_token
        
        latent = self.encoder(x_masked)
        pred = self.output_proj(latent)
        
        mask_float = mask.unsqueeze(-1).float()
        mse_per_window = ((pred - x) ** 2 * mask_float).sum(dim=1) / (mask_float.sum(dim=1) + 1e-9)
        mean_x = (x * mask_float).sum(dim=1, keepdim=True) / (mask_float.sum(dim=1, keepdim=True) + 1e-9)
        var_x_per_window = ((x - mean_x) ** 2 * mask_float).sum(dim=1) / (mask_float.sum(dim=1) + 1e-9)
        fvu_per_window = mse_per_window / (var_x_per_window + 1e-6)
        
        return mse_per_window.mean(), fvu_per_window.mean()

def load_and_split_data_audited(gcs_path, num_shards=8):
    print(f"[BLITZ-V3] Stratified Ingestion: Sampling {num_shards} shards across 2025 timeline...", flush=True)
    fs = gcsfs.GCSFileSystem()
    # 🚀 强制锁定 2025 年数据
    files = sorted(fs.glob(f"{gcs_path}/2025*.parquet"))
    if not files: raise ValueError(f"FATAL: No 2025 shards at {gcs_path}")
    
    step = max(1, len(files) // num_shards)
    sampled_files = files[::step][:num_shards]
    
    shard_info = []
    total_rows = 0
    columns_to_load = ['price_change', 'srl_residual', 'epiplexity']
    
    import pyarrow.parquet as pq
    for f in sampled_files:
        with fs.open(f) as pf:
            rows = pq.read_metadata(pf).num_rows
            shard_info.append((f, rows))
            total_rows += rows
            
    print(f"[BLITZ-V3] Total rows: {total_rows:,}. Perfect for 16GB RAM.", flush=True)
    full_data = torch.empty((total_rows, 3), dtype=torch.float32)
    
    split_idx_shard = int(len(sampled_files) * 0.8)
    train_shards = shard_info[:split_idx_shard]
    
    print(f"[BLITZ-V3] Phase 1: Calculating Stats (Float64 Precision)...", flush=True)
    sum_val = np.zeros(3, dtype=np.float64)
    sum_sq = np.zeros(3, dtype=np.float64)
    train_rows = 0
    for f, rows in train_shards:
        df = pd.read_parquet(f"gs://{f}", columns=columns_to_load, engine='pyarrow')
        arr = df.values.astype(np.float64)
        sum_val += arr.sum(axis=0)
        sum_sq += (arr**2).sum(axis=0)
        train_rows += rows
        del df, arr; gc.collect()
        
    mean = sum_val / train_rows
    std = np.sqrt((sum_sq / train_rows) - mean**2) + 1e-8
    
    print(f"[BLITZ-V3] Phase 2: Loading & Normalizing...", flush=True)
    current_pos = 0
    mean_t = torch.from_numpy(mean).float()
    std_t = torch.from_numpy(std).float()
    for f, rows in shard_info:
        df = pd.read_parquet(f"gs://{f}", columns=columns_to_load, engine='pyarrow')
        shard_tensor = torch.from_numpy(df.values).float()
        shard_tensor.sub_(mean_t).div_(std_t) 
        full_data[current_pos : current_pos + rows] = shard_tensor
        current_pos += rows
        del df, shard_tensor; gc.collect()
        
    train_data, val_data = full_data[:train_rows], full_data[train_rows:]
    train_data.share_memory_()
    val_data.share_memory_()
    return train_data, val_data

class EpiplexityShardDataset(Dataset):
    def __init__(self, data_tensor, seq_len, stride):
        self.data, self.seq_len, self.stride = data_tensor, seq_len, stride
        self.span = (seq_len - 1) * stride + 1
    def __len__(self): return len(self.data) - self.span
    def __getitem__(self, idx): return self.data[idx : idx + self.span : self.stride].clone()

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
    train_data, val_data = load_and_split_data_audited(args.gcs_input, num_shards=8)
    
    train_loader = DataLoader(EpiplexityShardDataset(train_data, args.seq_len, args.stride), batch_size=args.micro_batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(EpiplexityShardDataset(val_data, args.seq_len, args.stride), batch_size=args.micro_batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    model = EpiplexityMAE(seq_len=args.seq_len).to(device)
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
                print(f"🚨 [FATAL] NaN detected. Suiciding.", flush=True)
                hpt_client.report_hyperparameter_tuning_metric(hyperparameter_metric_tag='val_fvu', metric_value=9999.0, global_step=logical_step)
                sys.exit(0)
                
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if logical_step % args.report_freq == 0:
            val_fvu = fast_validate(model, val_iter, device, max_batches=50)
            print(f"⚡ [SONAR] Step {logical_step:04d}/{args.max_steps} | Val FVU: {val_fvu:.5f}", flush=True)
            sys.stdout.flush()
            hpt_client.report_hyperparameter_tuning_metric(hyperparameter_metric_tag='val_fvu', metric_value=val_fvu, global_step=logical_step)
            
            # 🪓 [架构师修正版拔管协议]：在第 1000 步时，只有 FVU >= 1.15（证明完全不收敛）才斩杀。
            # 之前的 0.99 过于残酷，误杀了 1.009 的黄金种子。
            if logical_step >= 1000 and val_fvu >= 1.15:
                print(f"☠️ [EXECUTION] Noise detected (FVU {val_fvu:.4f} >= 1.15). Suiciding.", flush=True)
                hpt_client.report_hyperparameter_tuning_metric(hyperparameter_metric_tag='val_fvu', metric_value=9999.0, global_step=args.max_steps)
                sys.exit(0)

if __name__ == "__main__":
    try: torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError: pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--gcs_input", type=str, required=True)
    parser.add_argument("--logical_batch_size", type=int, default=4096)
    parser.add_argument("--micro_batch_size", type=int, default=512) 
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--report_freq", type=int, default=500)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=400)
    args, _ = parser.parse_known_args()
    forge_compressor(args)
