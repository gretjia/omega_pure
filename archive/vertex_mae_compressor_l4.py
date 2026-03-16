"""
THE OMEGA PROTOCOL: THE KOLMOGOROV COMPRESSOR (L4 WOLFPACK EDITION)
Paradigm: Scale-Invariant Epiplexity Discovery
Discipline: OOM KILLER IMMUNITY. ASYNC DATALOADER. STRICT HEARTBEAT.
"""
import os
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

# =====================================================================
# MODULE 1: THE TURING MACHINE (MAE ARCHITECTURE)
# =====================================================================
class EpiplexityMAE(nn.Module):
    def __init__(self, feature_dim=3, seq_len=64, embed_dim=64, num_heads=4, depth=4, mask_ratio=0.70):
        super().__init__()
        self.seq_len = seq_len
        self.mask_ratio = mask_ratio
        
        self.input_proj = nn.Linear(feature_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True, activation='gelu'
        )
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
        
        # Local MSE
        mse_per_window = ((pred - x) ** 2 * mask_float).sum(dim=1) / (mask_float.sum(dim=1) + 1e-9)
        
        # Local Variance
        mean_x = (x * mask_float).sum(dim=1, keepdim=True) / (mask_float.sum(dim=1, keepdim=True) + 1e-9)
        var_x_per_window = ((x - mean_x) ** 2 * mask_float).sum(dim=1) / (mask_float.sum(dim=1) + 1e-9)
        
        # Fraction of Variance Unexplained (FVU) - 无量纲防御
        fvu_per_window = mse_per_window / (var_x_per_window + 1e-6)
        
        scale_invariant_loss = fvu_per_window.mean()
        raw_mse = mse_per_window.mean()
        
        return raw_mse, scale_invariant_loss

# =====================================================================
# MODULE 2: OOM-SAFE LOADER & READ-ONLY DATASET
# =====================================================================
def load_and_split_data(gcs_path, limit_files=100):
    print(f"[SYSTEM] Connecting to GCS: {gcs_path}", flush=True)
    fs = gcsfs.GCSFileSystem()
    files = sorted(fs.glob(f"{gcs_path}/*.parquet"))
    if not files: raise ValueError(f"FATAL: No shards at {gcs_path}")
        
    files = files[:limit_files]
    print(f"[SYSTEM] Ingesting {len(files)} shards for HPO...", flush=True)
    
    columns_to_load = ['price_change', 'srl_residual', 'epiplexity']
    df_list = []
    for f in files:
        df = pd.read_parquet(f"gs://{f}", columns=columns_to_load, engine='pyarrow')
        df_list.append(torch.tensor(df.values, dtype=torch.float32))
        
    full_data = torch.cat(df_list, dim=0)
    del df_list
    gc.collect() # 强行回收内存碎片，防 OOM
    
    split_idx = int(len(full_data) * 0.8)
    train_data = full_data[:split_idx]
    val_data = full_data[split_idx:]
    
    # 🚀 [架构师级内存控制]：必须使用 In-place 操作(.sub_/.div_)，避免内存暴增 50%！
    mean = torch.mean(train_data, dim=0)
    std = torch.std(train_data, dim=0) + 1e-8
    
    train_data.sub_(mean).div_(std)
    val_data.sub_(mean).div_(std)
    
    # 🚀 [架构师防 CoW 核爆]：强行锁入共享内存，防 Linux Fork 复制引发 OOM Killer
    train_data.share_memory_()
    val_data.share_memory_()
    
    print(f"[SYSTEM] Memory forged successfully. Shape: {train_data.shape}", flush=True)
    return train_data, val_data

class EpiplexityShardDataset(Dataset):
    def __init__(self, data_tensor, seq_len, stride):
        self.data = data_tensor
        self.seq_len = seq_len
        self.stride = stride
        self.span = (seq_len - 1) * stride + 1

    def __len__(self):
        return len(self.data) - self.span

    def __getitem__(self, idx):
        return self.data[idx : idx + self.span : self.stride]

# =====================================================================
# MODULE 3: SINGLE-GPU, HEARTBEAT ENGINE
# =====================================================================
def forge_compressor(args):
    # 🚀 毒点剔除：强行单卡，彻底抛弃 DataParallel 毒瘤
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hpt_client = hypertune.HyperTune()
    
    print(f"[SYSTEM] Vizier L4 Probe | Seq: {args.seq_len} | Stride: {args.stride} | Device: {device}", flush=True)
    train_data, val_data = load_and_split_data(args.gcs_input, limit_files=args.lookback_days)
    
    train_dataset = EpiplexityShardDataset(train_data, args.seq_len, args.stride)
    val_dataset = EpiplexityShardDataset(val_data, args.seq_len, args.stride)
    
    # 🚀 工业级异步加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        drop_last=True,
        num_workers=4,           # 解放主进程死锁
        pin_memory=True,         # 锁页内存直通 GPU
        prefetch_factor=2,       # 预取防 GPU 饥饿
        persistent_workers=True  # 防每 Epoch 销毁开销
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    model = EpiplexityMAE(seq_len=args.seq_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    for epoch in range(args.epochs):
        model.train()
        for i, batch in enumerate(train_loader):
            # non_blocking=True 配合 pin_memory 才能达成极速异步
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            _, fvu_loss = model(batch)
            fvu_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # 🚀 心跳探针：强行穿透 GCP 日志缓冲
            if i % 500 == 0:
                print(f"❤️ [HEARTBEAT] Epoch {epoch+1:03d} | Batch {i:05d} | FVU Loss: {fvu_loss.item():.4f}", flush=True)
                sys.stdout.flush()
                
        # VALIDATION (Source of Truth for Vizier)
        model.eval()
        val_fvu_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device, non_blocking=True)
                _, batch_fvu = model(batch)
                val_fvu_total += batch_fvu.item()
                
        avg_val_fvu = val_fvu_total / len(val_loader) if len(val_loader) > 0 else float('inf')
        print(f"✅ Epoch {epoch+1:03d} COMPLETE | Val FVU: {avg_val_fvu:.5f}", flush=True)
        sys.stdout.flush()
        
        # 上报给贝叶斯引擎
        hpt_client.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag='val_fvu',
            metric_value=avg_val_fvu,
            global_step=epoch
        )

if __name__ == "__main__":
    # 防 MacOS/Linux 多进程启动冲突
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--gcs_input", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=10)
    
    # 动态注入
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=400)
    parser.add_argument("--lookback_days", type=int, default=30)
    
    args, _ = parser.parse_known_args()
    forge_compressor(args)