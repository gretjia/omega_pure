"""
THE OMEGA PROTOCOL: THE KOLMOGOROV COMPRESSOR (BLITZKRIEG EDITION)
Paradigm: Successive Halving & Fast Exploration
Discipline: DATA TRUNCATION. MICRO-SONAR VALIDATION. RUTHLESS SELF-DESTRUCT.
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

# 解锁 Ada 架构极速算力
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

# =====================================================================
# MODULE 2: LIGHTNING DATA INGESTION
# =====================================================================
def load_and_split_data(gcs_path, limit_files=30): 
    # 🚀 极速装载：闪电战只需加载前 30 个碎片，瞬间完成 I/O！
    print(f"[BLITZ] Ingesting {limit_files} shards for rapid parameter discovery...", flush=True)
    fs = gcsfs.GCSFileSystem()
    files = sorted(fs.glob(f"{gcs_path}/*.parquet"))
    if not files: raise ValueError(f"FATAL: No shards at {gcs_path}")
        
    files = files[:limit_files]
    columns_to_load = ['price_change', 'srl_residual', 'epiplexity']
    df_list = []
    for f in files:
        df = pd.read_parquet(f"gs://{f}", columns=columns_to_load, engine='pyarrow')
        df_list.append(torch.tensor(df.values, dtype=torch.float32))
        
    full_data = torch.cat(df_list, dim=0)
    del df_list; gc.collect() 
    
    split_idx = int(len(full_data) * 0.8)
    train_data, val_data = full_data[:split_idx], full_data[split_idx:]
    
    mean, std = torch.mean(train_data, dim=0), torch.std(train_data, dim=0) + 1e-8
    train_data.sub_(mean).div_(std)
    val_data.sub_(mean).div_(std)
    
    train_data.share_memory_()
    val_data.share_memory_()
    return train_data, val_data

class EpiplexityShardDataset(Dataset):
    def __init__(self, data_tensor, seq_len, stride):
        self.data, self.seq_len, self.stride = data_tensor, seq_len, stride
        self.span = (seq_len - 1) * stride + 1
    def __len__(self): return len(self.data) - self.span
    def __getitem__(self, idx): return self.data[idx : idx + self.span : self.stride].clone()

# =====================================================================
# MODULE 3: THE MICRO-SONAR VALIDATION
# =====================================================================
@torch.no_grad()
def fast_validate(model, val_iter, device, max_batches=50):
    """🚀 架构师防死锁补丁：每次只抽样 50 个 Batch 进行极速验证，耗时 < 2 秒"""
    model.eval()
    val_fvu_total = 0.0
    count = 0
    for _ in range(max_batches):
        try:
            batch = next(val_iter)
        except StopIteration:
            break
        batch = batch.to(device, non_blocking=True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            _, batch_fvu = model(batch)
        val_fvu_total += batch_fvu.item()
        count += 1
        
    model.train() 
    return val_fvu_total / max(1, count)

# =====================================================================
# MODULE 4: THE BLITZKRIEG ENGINE
# =====================================================================
def forge_compressor(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hpt_client = hypertune.HyperTune()
    
    accum_steps = max(1, args.logical_batch_size // args.micro_batch_size)
    print(f"[BLITZ] Node Awake | Seq: {args.seq_len} | Stride: {args.stride}", flush=True)
    
    train_data, val_data = load_and_split_data(args.gcs_input, limit_files=30)
    
    train_loader = DataLoader(EpiplexityShardDataset(train_data, args.seq_len, args.stride), batch_size=args.micro_batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(EpiplexityShardDataset(val_data, args.seq_len, args.stride), batch_size=args.micro_batch_size, shuffle=True, num_workers=4, pin_memory=True) 
    
    model = EpiplexityMAE(seq_len=args.seq_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # 构造无限迭代器，废除传统 Epoch
    def infinite_loader(dl):
        while True:
            for b in dl: yield b
            
    train_iter = infinite_loader(train_loader)
    val_iter = infinite_loader(val_loader)
    
    model.train()
    
    # 🚀 直接按逻辑更新步数 (Logical Steps) 迭代
    for logical_step in range(1, args.max_steps + 1):
        optimizer.zero_grad()
        
        # 内部执行微批次梯度累加
        for _ in range(accum_steps):
            try:
                batch = next(train_iter).to(device, non_blocking=True)
            except StopIteration:
                break
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                _, fvu_loss = model(batch)
                loss = fvu_loss / accum_steps
            loss.backward()
            
            # 🩸 [基因自毁] NaN 物理坍塌斩首
            if torch.isnan(fvu_loss) or torch.isinf(fvu_loss):
                print(f"🚨 [FATAL] NaN detected at step {logical_step}. Suiciding to free Quota.", flush=True)
                hpt_client.report_hyperparameter_tuning_metric(hyperparameter_metric_tag='val_fvu', metric_value=9999.0, global_step=logical_step)
                sys.exit(0)
                
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # ⚡ 极速声呐验证与拔管协议
        if logical_step % args.report_freq == 0:
            val_fvu = fast_validate(model, val_iter, device, max_batches=50)
            print(f"⚡ [SONAR] Step {logical_step:04d}/{args.max_steps} | Val FVU: {val_fvu:.5f}", flush=True)
            sys.stdout.flush()
            
            hpt_client.report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag='val_fvu', metric_value=val_fvu, global_step=logical_step
            )
            
            # 🪓 [塔勒布式拔管]：在第 1000 步时，如果 FVU 依然 >= 0.99，证明模型毫无压缩能力，当场处决！
            if logical_step >= 1000 and val_fvu >= 0.99:
                print(f"☠️ [EXECUTION] Weak config (FVU {val_fvu:.4f} >= 0.99). Pulling the plug.", flush=True)
                hpt_client.report_hyperparameter_tuning_metric(hyperparameter_metric_tag='val_fvu', metric_value=9999.0, global_step=args.max_steps)
                sys.exit(0)

    print(f"✅ Trial Complete. Maximum steps reached.", flush=True)

if __name__ == "__main__":
    try: torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError: pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--gcs_input", type=str, required=True)
    parser.add_argument("--logical_batch_size", type=int, default=4096)
    parser.add_argument("--micro_batch_size", type=int, default=512) 
    
    # 闪电战核心：最多只跑 3000 步，每 500 步上报一次
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--report_freq", type=int, default=500)
    
    # Vizier 动态注入
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=400)
    
    args, _ = parser.parse_known_args()
    forge_compressor(args)
