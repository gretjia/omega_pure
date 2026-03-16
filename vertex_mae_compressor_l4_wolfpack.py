"""
THE OMEGA PROTOCOL: THE KOLMOGOROV COMPRESSOR (L4 WOLFPACK EDITION)
Paradigm: Scale-Invariant Epiplexity Discovery
Discipline: MICRO-BATCHING + BFLOAT16 + IPC ZERO-LEAKAGE.
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

# 🚀 [架构师指令] 解锁 Ada 架构 TF32 极速算力
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# =====================================================================
# MODULE 1: THE TURING MACHINE
# =====================================================================
class EpiplexityMAE(nn.Module):
    def __init__(self, feature_dim=3, seq_len=64, embed_dim=64, num_heads=4, depth=4, mask_ratio=0.70):
        super().__init__()
        self.seq_len = seq_len
        self.mask_ratio = mask_ratio
        
        self.input_proj = nn.Linear(feature_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # PyTorch 2.x 会在此自动路由至 FlashAttention 节约显存
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
        
        # 绝对物理约束：FVU (Fraction of Variance Unexplained)
        mse_per_window = ((pred - x) ** 2 * mask_float).sum(dim=1) / (mask_float.sum(dim=1) + 1e-9)
        mean_x = (x * mask_float).sum(dim=1, keepdim=True) / (mask_float.sum(dim=1, keepdim=True) + 1e-9)
        var_x_per_window = ((x - mean_x) ** 2 * mask_float).sum(dim=1) / (mask_float.sum(dim=1) + 1e-9)
        
        fvu_per_window = mse_per_window / (var_x_per_window + 1e-6)
        
        return mse_per_window.mean(), fvu_per_window.mean()

# =====================================================================
# MODULE 2: CPU RAM FORTRESS & OOM-SAFE LOADER
# =====================================================================
def load_and_split_data(gcs_path, limit_files=150):
    print(f"[SYSTEM] Establishing uplink to GCS: {gcs_path}", flush=True)
    fs = gcsfs.GCSFileSystem()
    files = sorted(fs.glob(f"{gcs_path}/*.parquet"))
    if not files: raise ValueError(f"FATAL: No shards at {gcs_path}")
        
    # 为了 HPO 的效率，我们载入前 150 个碎片，足以覆盖数千万行数据寻找物理常数
    files = files[:limit_files]
    print(f"[SYSTEM] Ingesting {len(files)} shards into 128GB RAM Fortress...", flush=True)
    
    columns_to_load = ['price_change', 'srl_residual', 'epiplexity']
    df_list = []
    for f in files:
        df = pd.read_parquet(f"gs://{f}", columns=columns_to_load, engine='pyarrow')
        df_list.append(torch.tensor(df.values, dtype=torch.float32))
        
    full_data = torch.cat(df_list, dim=0)
    del df_list
    gc.collect() 
    
    split_idx = int(len(full_data) * 0.8)
    train_data = full_data[:split_idx]
    val_data = full_data[split_idx:]
    
    # In-place 操作，防止 CPU RAM 翻倍
    mean = torch.mean(train_data, dim=0)
    std = torch.std(train_data, dim=0) + 1e-8
    train_data.sub_(mean).div_(std)
    val_data.sub_(mean).div_(std)
    
    # 锁页到共享内存，防止 Linux Fork 复制引发死机
    train_data.share_memory_()
    val_data.share_memory_()
    
    print(f"[SYSTEM] Memory forged successfully. Shape: {train_data.shape} (RAM usage: ~{train_data.element_size() * train_data.nelement() / 1e9:.2f} GB)", flush=True)
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
        # 🚀 [架构师防漏救命补丁]: 必须加 .clone()! 
        # 彻底斩断 IPC 通信中的海量 Storage 引用，防 CPU 内存炸毁！
        return self.data[idx : idx + self.span : self.stride].clone()

# =====================================================================
# MODULE 3: THE VRAM-DEFENDED TRAINING ENGINE
# =====================================================================
def forge_compressor(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hpt_client = hypertune.HyperTune()
    
    # 🚀 [核心物理防线] 逻辑批次与物理显存解绑
    accum_steps = max(1, args.logical_batch_size // args.micro_batch_size)
    print(f"[SYSTEM] L4 Probe | Seq: {args.seq_len} | Stride: {args.stride} | Lookback Days: {args.lookback_days}", flush=True)
    print(f"[SYSTEM] VRAM Armor Engaged: Logical Batch {args.logical_batch_size} -> {accum_steps} steps of Micro-Batch {args.micro_batch_size}", flush=True)
    
    train_data, val_data = load_and_split_data(args.gcs_input, limit_files=args.lookback_days)
    
    train_dataset = EpiplexityShardDataset(train_data, args.seq_len, args.stride)
    val_dataset = EpiplexityShardDataset(val_data, args.seq_len, args.stride)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.micro_batch_size, shuffle=True, 
        drop_last=True, num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.micro_batch_size, shuffle=False, 
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    
    model = EpiplexityMAE(seq_len=args.seq_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        
        for i, batch in enumerate(train_loader):
            batch = batch.to(device, non_blocking=True)
            
            # 🚀 激活 BFloat16 混合精度：计算快一倍，显存再省一半！
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                _, fvu_loss = model(batch)
                # Loss 必须除以累加步数，以保证梯度的数学等价性
                loss = fvu_loss / accum_steps
                
            loss.backward()
            
            # 🚀 梯度累积：当达到逻辑批次大小时，再执行一次权重更新
            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                logical_step = (i + 1) // accum_steps
                
                # 🚀 [架构师防漏救命补丁：NaN 基因自毁与止损协议]
                if torch.isnan(fvu_loss) or torch.isinf(fvu_loss):
                    print(f"💀 [AUTO-KILL] Trial has diverged (NaN/Inf detected at Epoch {epoch+1} Step {logical_step}). Initiating self-destruct to release L4 GPU...", flush=True)
                    sys.stdout.flush()
                    # 上报一个极差的分数给贝叶斯引擎，然后直接拔管
                    hpt_client.report_hyperparameter_tuning_metric(
                        hyperparameter_metric_tag='val_fvu', 
                        metric_value=9999.0, # 极其巨大的惩罚值
                        global_step=epoch
                    )
                    sys.exit(0) # 优雅退出容器，Vertex 调度器会立刻把卡分配给下一个排队的 Trial
                
                if logical_step % 100 == 0:
                    print(f"❤️ [HEARTBEAT] Epoch {epoch+1:02d} | Step {logical_step:04d} | FVU Loss: {fvu_loss.item():.4f}", flush=True)
                    sys.stdout.flush()
                
        # VALIDATION (The Source of Truth)
        model.eval()
        val_fvu_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device, non_blocking=True)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    _, batch_fvu = model(batch)
                val_fvu_total += batch_fvu.item()
                
        avg_val_fvu = val_fvu_total / len(val_loader) if len(val_loader) > 0 else float('inf')
        print(f"✅ Epoch {epoch+1:02d} COMPLETE | Val FVU: {avg_val_fvu:.5f}", flush=True)
        sys.stdout.flush()
        
        hpt_client.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag='val_fvu', metric_value=avg_val_fvu, global_step=epoch
        )

if __name__ == "__main__":
    try: torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError: pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--gcs_input", type=str, required=True)
    parser.add_argument("--lookback_days", type=int, default=30) 
    
    # 核心：外界传入庞大的逻辑 Batch (4096)，物理执行锁定在安全的 512 微批次！
    parser.add_argument("--logical_batch_size", type=int, default=4096)
    parser.add_argument("--micro_batch_size", type=int, default=512) 
    
    parser.add_argument("--epochs", type=int, default=10)
    
    # Vizier 动态注入
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=400)
    
    args, _ = parser.parse_known_args()
    forge_compressor(args)