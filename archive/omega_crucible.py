"""
THE OMEGA PROTOCOL: THE TALEBIAN ORACLE & DEEP-SPACE BACKTESTER
Execution Target: Mac Studio (Apple Silicon MPS / CPU) or AMD ROCm
Discipline: ZERO-COPY TENSOR WINDOWS. STRICT T+1 A-SHARE LOGIC. NO OOP BLOAT.
"""

import os
import torch
import numpy as np
import pandas as pd
from glob import glob
from vertex_mae_compressor import EpiplexityMAE # 导入 Step 3 的模型架构

# =====================================================================
# MODULE 1: THE TALEBIAN ORACLE (SIGNAL GENERATOR)
# =====================================================================

class TalebianOracle:
    def __init__(self, model_path: str, z_score_threshold: float = -3.5, seq_len: int = 64):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"[ORACLE] Awakening on Silicon: {self.device}")
        
        # 1. 唤醒 Vertex 锻造的基线
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model = EpiplexityMAE(seq_len=seq_len).to(self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval() # 绝对锁定，禁止反向传播
        
        self.mean = torch.tensor(checkpoint['mean'], device=self.device)
        self.std = torch.tensor(checkpoint['std'], device=self.device)
        
        # 2. 设定绝对物理阈值
        self.baseline_loss = checkpoint['baseline_loss']
        
        # 假定验证集测出的误差标准差约为基线的 15% (需根据 2024 年数据实际测量后修改)
        self.loss_std = self.baseline_loss * 0.15 
        self.kill_threshold = self.baseline_loss + (z_score_threshold * self.loss_std)
        self.seq_len = seq_len
        
        print(f"[ORACLE] High-Entropy Baseline: {self.baseline_loss:.6f}")
        print(f"[ORACLE] Kill Switch Threshold (Loss <): {self.kill_threshold:.6f}")

    @torch.no_grad()
    def batch_scan_event_horizon(self, tensor_batch: torch.Tensor):
        """
        [PURE TENSOR INFERENCE]
        批量计算数万个滑动窗口的重构误差，寻找可压缩的低熵奇点。
        """
        x_norm = (tensor_batch.to(self.device) - self.mean) / (self.std + 1e-8)
        
        # 将 70% 遮蔽，强迫重构
        _, pred = self.model(x_norm)
        
        # 计算每个窗口的 MSE Loss (Batch, )
        mse_per_window = torch.mean((pred - x_norm)**2, dim=[1, 2])
        return mse_per_window.cpu().numpy()

# =====================================================================
# MODULE 2: VECTORIZED CRUCIBLE BACKTESTER (T+1 Mechanics)
# =====================================================================

def compute_asymmetry_payoff(trade_log: list):
    """塔勒布非对称收益评估：只看极端的赔率，无视平庸的胜率"""
    if not trade_log:
        print("\n[VERDICT] No Executions. Market remained in high-entropy retail noise.")
        return
        
    df = pd.DataFrame(trade_log)
    win_trades = df[df['pnl_pct'] > 0]['pnl_pct']
    loss_trades = df[df['pnl_pct'] <= 0]['pnl_pct']
    
    win_rate = len(win_trades) / len(df)
    avg_win = win_trades.mean() if len(win_trades) > 0 else 0
    avg_loss = abs(loss_trades.mean()) if len(loss_trades) > 0 else 1e-9
    asymmetry_ratio = avg_win / avg_loss
    
    print("\n" + "="*50)
    print("🩸 THE CRUCIBLE VERDICT (OUT-OF-SAMPLE)")
    print("="*50)
    print(f"Total Hunt Executions: {len(df)}")
    print(f"Win Rate:              {win_rate*100:.2f}% (Taleb says: Irrelevant)")
    print(f"Average Win:           +{avg_win*100:.2f}%")
    print(f"Average Loss:          -{avg_loss*100:.2f}%")
    print(f"Asymmetry Ratio:       {asymmetry_ratio:.2f} (Target: > 3.0)")
    print(f"Net Strategy Return:   {df['pnl_pct'].sum()*100:.2f}% (Uncompounded)")
    print("="*50)

def stream_blind_backtest(data_dir: str, oracle: TalebianOracle, dilation_stride: int = 400, holding_ticks: int = 48000):
    """
    极速流式回测引擎 (波段视野优化版)。
    dilation_stride = 400: 必须与 Vertex 训练时的 stride 绝对一致！
    holding_ticks = 48000: 塔勒布式大波段持仓周期 (约 10 个交易日/2周)。
    """
    parquet_files = sorted(glob(f"{data_dir}/*.parquet"))
    if not parquet_files:
        raise ValueError(f"FATAL: No shards found in {data_dir}. Run Step 2 Forge first.")

    print(f"[CRUCIBLE] Ingesting {len(parquet_files)} temporal shards for absolute blind test...")
    trade_log = []

    # 物理摩擦成本 (假设单边印花税万五 + 佣金万一 + 极端滑点万四 = 千分之一)
    FRICTION_COST = 0.0010 

    for file in parquet_files:
        df = pd.read_parquet(file)

        # 提取 Step 2 锻造的特征 (必须是 float32)
        # 假设 df 列名: ['price', 'order_flow', 'price_change', 'srl_residual', 'epiplexity']
        features = df[['price_change', 'srl_residual', 'epiplexity']].values.astype(np.float32)
        prices = df['price'].values
        order_flows = df['order_flow'].values # 用于判断主力是在吸筹还是派发

        N = len(features)
        seq_len = oracle.seq_len
        span = (seq_len - 1) * dilation_stride + 1  # 宏观视野物理跨度

        if N <= span: continue

        feature_tensor = torch.tensor(features)

        # 🚀 [核心架构黑魔法 2.0：Dilated Zero-Copy Sliding Window]
        # 获取张量底层的物理内存步长
        stride_n, stride_f = feature_tensor.stride()
        num_windows = N - span + 1

        # 利用 as_strided 直接修改底层内存映射！不复制 1 byte 数据！
        # 维度 0: 窗口每次向前滑动 1 个 Tick (确保不漏掉任何一秒的触发点)
        # 维度 1: 窗口内部的 64 个 Token，彼此间隔 dilation_stride 个 Tick (宏观视野)
        # 维度 2: 特征维度
        windows = torch.as_strided(
            feature_tensor, 
            size=(num_windows, seq_len, feature_tensor.shape[1]), 
            stride=(stride_n, stride_n * dilation_stride, stride_f)
        )

        # 批处理推理防 OOM (利用 Mac Studio / AMD 的显存带宽)
        batch_size = 4096
        all_losses = []
        for i in range(0, len(windows), batch_size):
            batch = windows[i : i + batch_size]
            losses = oracle.batch_scan_event_horizon(batch)
            all_losses.extend(losses)

        all_losses = np.array(all_losses)

        # 寻找“熵塌缩”奇点 (Loss 断崖式暴跌)
        anomaly_indices = np.where(all_losses < oracle.kill_threshold)[0]

        # 状态机：T+1 冷却锁定
        exit_idx = 0 

        for idx in anomaly_indices:
            if idx < exit_idx: continue # 还在持仓中，遵守 T+1 纪律，忽略新信 号

            # ⚠️ 关键修正：由于拉长了视野，真正探测到异常奇点的那一刻，
            # 是这整个宏观窗口走完的最后一笔 Tick，即 idx + span - 1
            absolute_end_idx = idx + span - 1

            # 计算这整整一周（span）内的净买卖力量！
            # 过滤高频的诱多假动作，看到的是主力宏观上的真实建仓意图
            net_flow = np.sum(order_flows[idx : absolute_end_idx + 1])

            # A 股做空受限，我们严格遵循 Via Negativa，只跟随主力做多，避开主 力派发
            if net_flow > 0:
                entry_price = prices[absolute_end_idx]
                exit_idx = min(absolute_end_idx + holding_ticks, N - 1)
                exit_price = prices[exit_idx]

                # 计算纯粹的非对称盈亏 (扣除双边摩擦成本)
                pnl = (exit_price - entry_price) / entry_price - (FRICTION_COST * 2)

                trade_log.append({
                    'shard': os.path.basename(file),
                    'loss_val': all_losses[idx],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_pct': pnl
                })
                # 打印击杀日志
                print(f"⚔️ [MACRO STRIKE] {os.path.basename(file)} | Window: ~5 Days | Return: {pnl*100:+.2f}%")

    compute_asymmetry_payoff(trade_log)

if __name__ == "__main__":
    # 替换为你在 GCP 炼丹下载回来的模型路径
    MODEL_PATH = "omega_mae_oracle.pth"
    
    # 填入经过你 AMD 机器 Step 2 处理过后的 2025 盲测数据集路径
    BLIND_TEST_DIR = "./base_matrix_shards_2025" 
    
    if os.path.exists(MODEL_PATH) and os.path.exists(BLIND_TEST_DIR):
        # 设定严苛的 -3.5 个标准差作为猎杀阈值
        oracle = TalebianOracle(model_path=MODEL_PATH, z_score_threshold=-3.5)
        stream_blind_backtest(data_dir=BLIND_TEST_DIR, oracle=oracle)
    else:
        print("[SYSTEM] 武器未就绪。请确保在 GCP 完成 Step 3，并在 AMD 完成 2025 数据 Forge。")