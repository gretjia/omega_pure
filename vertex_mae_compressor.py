"""
THE OMEGA PROTOCOL: THE KOLMOGOROV COMPRESSOR (HPO READY)
Paradigm: Scale-Invariant Epiplexity Discovery
Discipline: STRICT CHRONOLOGICAL SPLIT. LOCAL FVU OPTIMIZATION.
"""
import os
import argparse
import gc
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import gcsfs
import hypertune  # Google Cloud Vizier Reporter
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
        x_emb = self.input_proj(x)
        
        noise = torch.rand(B, T, device=x.device)
        mask = (noise < self.mask_ratio).bool()
        
        x_masked = x_emb.clone()
        x_masked[mask] = self.mask_token
        
        # [DATA LEAKAGE FIX]: Add pos_embed AFTER masking so masked tokens retain positional awareness
        x_masked = x_masked + self.pos_embed
        
        latent = self.encoder(x_masked)
        pred = self.output_proj(latent)
        
        mask_float = mask.unsqueeze(-1).float()
        
        # 1. 局部绝对重构误差 (Local MSE)
        mse_per_window = ((pred - x) ** 2 * mask_float).sum(dim=1) / (mask_float.sum(dim=1) + 1e-9)
        
        # 2. [核心物理修正]：计算每个窗口内被遮蔽部分的真实局部方差
        mean_x = (x * mask_float).sum(dim=1, keepdim=True) / (mask_float.sum(dim=1, keepdim=True) + 1e-9)
        var_x_per_window = ((x - mean_x) ** 2 * mask_float).sum(dim=1) / (mask_float.sum(dim=1) + 1e-9)
        
        # 3. 局部未解释方差比 (Fraction of Variance Unexplained - FVU)
        # 彻底粉碎 Vizier 偷懒选择极小 stride 的可能
        fvu_per_window = mse_per_window / (var_x_per_window + 1e-6)
        
        scale_invariant_loss = fvu_per_window.mean()
        raw_mse = mse_per_window.mean()
        
        return raw_mse, scale_invariant_loss

# =====================================================================
# MODULE 2: CHRONOLOGICAL OOM-SAFE LOADER
# =====================================================================
def load_and_split_data(gcs_path, limit_files=20):
    print(f"[SYSTEM] Connecting to GCS: {gcs_path}", flush=True)
    fs = gcsfs.GCSFileSystem()
    files = sorted(fs.glob(f"{gcs_path}/*.parquet")) # 必须排序保证时间连续
    if not files: raise ValueError(f"FATAL: No shards at {gcs_path}")
        
    # [物理防线：数据驱动]: 限制加载的天数不再由人类拍脑袋决定。
    # 它将作为 `lookback_days` 超参，由 Vizier 动态探索 (3天, 5天, 甚至 20天)。
    files = files[:limit_files]
    
    columns_to_load = ['price_change', 'srl_residual', 'epiplexity']
    df_list = []
    
    print(f"[SYSTEM] Streaming {len(files)} shards into Memory...", flush=True)
    for i, f in enumerate(files):
        print(f"  -> Loading shard {i+1}/{len(files)}: {f}", flush=True)
        # Using pyarrow engine and directly converting columns to float32 numpy arrays before tensor conversion
        # to avoid Pandas memory bloat/fragmentation in container
        df = pd.read_parquet(f"gs://{f}", columns=columns_to_load, engine='pyarrow')
        tensor_data = torch.tensor(df[columns_to_load].values.astype(np.float32))
        df_list.append(tensor_data)
        del df
        
    print(f"[SYSTEM] Shards loaded. Concatenating tensors...", flush=True)
    full_data = torch.cat(df_list, dim=0)
    del df_list
    gc.collect()
    
    print(f"[SYSTEM] Tensors concatenated. Shape: {full_data.shape}. Splitting...", flush=True)
    # 严格的物理时间轴切割：前 80% 训练，后 20% 作为绝对盲测的验证集
    split_idx = int(len(full_data) * 0.8)
    train_data = full_data[:split_idx]
    val_data = full_data[split_idx:]
    
    print(f"[SYSTEM] Computing Z-Score (Mean/Std)...", flush=True)
    # Z-Score 必须且只能从 Train 计算，防止未来数据泄露
    mean = torch.mean(train_data, dim=0)
    std = torch.std(train_data, dim=0) + 1e-8
    
    print(f"[SYSTEM] Normalizing train_data...", flush=True)
    train_data = (train_data - mean) / std
    print(f"[SYSTEM] Normalizing val_data...", flush=True)
    val_data = (val_data - mean) / std
    print(f"[SYSTEM] Data ready.", flush=True)
    
    return train_data, val_data

class EpiplexityShardDataset(Dataset):
    def __init__(self, data_tensor, seq_len, stride):
        self.data = data_tensor
        self.seq_len = seq_len
        self.stride = stride
        self.span = (seq_len - 1) * stride + 1
        
        if len(self.data) <= self.span:
            raise ValueError(f"Dataset too small for Seq:{seq_len} and Stride:{stride}")

    def __len__(self):
        return len(self.data) - self.span

    def __getitem__(self, idx):
        return self.data[idx : idx + self.span : self.stride]

# =====================================================================
# MODULE 3: BAYESIAN HPO ENGINE
# =====================================================================
def forge_compressor(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hpt_client = hypertune.HyperTune()
    
    print(f"[SYSTEM] Vizier Probe | Seq: {args.seq_len} | Stride: {args.stride} | Lookback Days: {args.lookback_days}")
    train_data, val_data = load_and_split_data(args.gcs_input, limit_files=args.lookback_days)
    
    train_dataset = EpiplexityShardDataset(train_data, args.seq_len, args.stride)
    val_dataset = EpiplexityShardDataset(val_data, args.seq_len, args.stride)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = EpiplexityMAE(seq_len=args.seq_len).to(device)
    
    # ---------------------------------------------------------
    # MULTI-GPU IGNITION (DataParallel for 8x A100/L4)
    # ---------------------------------------------------------
    if torch.cuda.device_count() > 1:
        print(f"[SYSTEM] Detected {torch.cuda.device_count()} GPUs. Activating DataParallel...", flush=True)
        model = nn.DataParallel(model)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    for epoch in range(args.epochs):
        # TRAIN
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            _, scale_invariant_loss = model(batch)
            
            if scale_invariant_loss.dim() > 0:
                scale_invariant_loss = scale_invariant_loss.mean()
                
            # [FVU OPTIMIZATION FIX]: Backpropagate the FVU (Scale-Invariant Loss), not MSE
            scale_invariant_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        # VALIDATION (The Source of Truth)
        model.eval()
        val_fvu_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                _, batch_fvu = model(batch)
                
                if batch_fvu.dim() > 0:
                     batch_fvu = batch_fvu.mean()
                     
                val_fvu_total += batch_fvu.item()
                
        avg_val_fvu = val_fvu_total / len(val_loader) if len(val_loader) > 0 else float('inf')
        print(f"Epoch {epoch+1:03d} | Scale-Invariant Val FVU: {avg_val_fvu:.5f}", flush=True)
        
        # 绝对核心：向 Vizier 汇报的是无量纲化 FVU！
        hpt_client.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag='val_fvu',
            metric_value=avg_val_fvu,
            global_step=epoch
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gcs_input", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=10)
    
    # 由 Google Vizier 动态强行注入的常数
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=400)
    parser.add_argument("--lookback_days", type=int, default=5)
    
    args, _ = parser.parse_known_args()
    forge_compressor(args)
