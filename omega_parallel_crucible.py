import os
import torch
import numpy as np
import pandas as pd
from glob import glob
from scipy.stats import zscore
from concurrent.futures import ProcessPoolExecutor, as_completed

class TopoRadar:
    def __init__(self, model_path, days, ticks_per_day, z_score_trigger=-3.0):
        self.device = torch.device("cpu") 
        self.days = days
        self.ticks_per_day = ticks_per_day
        self.physical_ticks = 4800
        self.intraday_stride = self.physical_ticks // ticks_per_day
        self.intraday_span = (ticks_per_day - 1) * self.intraday_stride + 1
        self.total_span = (days - 1) * self.physical_ticks + self.intraday_span
        self.z_score_trigger = z_score_trigger
        
        self.mean = torch.zeros(3, device=self.device).view(1, 3, 1, 1)
        self.std = torch.ones(3, device=self.device).view(1, 3, 1, 1)

    @torch.no_grad()
    def extract_daily_2d_matrices(self, tensor_1d):
        N = len(tensor_1d)
        num_full_days = N // self.physical_ticks
        if num_full_days < self.days: return None, None
            
        windows_2d = []
        end_indices = []
        
        for current_day in range(self.days, num_full_days + 1):
            start_day = current_day - self.days
            start_tick = start_day * self.physical_ticks
            
            day_slices = []
            for d in range(self.days):
                d_start = start_tick + d * self.physical_ticks
                day_slices.append(tensor_1d[d_start : d_start + self.ticks_per_day * self.intraday_stride : self.intraday_stride])
                
            windows_2d.append(torch.stack(day_slices).permute(2, 0, 1))
            end_indices.append(current_day * self.physical_ticks - 1)
            
        return torch.stack(windows_2d), np.array(end_indices)

    @torch.no_grad()
    def scan_ticker(self, features):
        feature_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
        windows_2d, end_indices = self.extract_daily_2d_matrices(feature_tensor)
        if windows_2d is None: return None, None
        windows_2d = (windows_2d - self.mean) / (self.std + 1e-8)
        fvu_scores = np.random.normal(loc=1.0, scale=0.03, size=len(windows_2d)) # MOCK
        return fvu_scores, end_indices

def backtest_single_stock(args):
    file_path, oracle_config = args
    ticker_name = os.path.basename(file_path).split('.')[0]
    radar = TopoRadar(**oracle_config)
    
    try: df = pd.read_parquet(file_path)
    except Exception: return []
        
    features = df[['price_change', 'srl_residual', 'epiplexity']].values
    prices = df['price'].values
    order_flows = df['order_flow'].values
    
    fvu_scores, end_indices = radar.scan_ticker(features)
    if fvu_scores is None or len(fvu_scores) < 30: return []
        
    fvu_series = pd.Series(fvu_scores)
    rolling_mean = fvu_series.rolling(window=30, min_periods=10).mean()
    rolling_std = fvu_series.rolling(window=30, min_periods=10).std()
    z_scores = (fvu_series - rolling_mean) / (rolling_std + 1e-8)
    
    anomalies = np.where(z_scores < radar.z_score_trigger)[0]
    trade_log = []
    cooldown_tick = 0
    FRICTION = 0.0015
    HOLDING_TICKS = radar.days * radar.physical_ticks
    
    for i in anomalies:
        if pd.isna(z_scores[i]): continue
        abs_tick = end_indices[i]
        if abs_tick < cooldown_tick: continue
        
        start_tick = abs_tick - (radar.days * radar.physical_ticks) + 1
        net_flow = np.sum(order_flows[max(0, start_tick) : abs_tick + 1])
        
        if net_flow > 0:
            entry_price = prices[abs_tick]
            exit_tick = min(abs_tick + HOLDING_TICKS, len(prices) - 1)
            exit_price = prices[exit_tick]
            pnl = (exit_price - entry_price) / entry_price - (FRICTION * 2)
            
            path_prices = prices[abs_tick : exit_tick + 1]
            max_up = (np.max(path_prices) - entry_price) / entry_price
            max_down = (np.min(path_prices) - entry_price) / entry_price
            
            trade_log.append({
                'ticker': ticker_name,
                'signal_tick': abs_tick,
                'fvu': fvu_scores[i],
                'fvu_zscore': z_scores[i],
                'pnl_pct': pnl,
                'max_up': max_up,
                'max_down': max_down
            })
            cooldown_tick = exit_tick
    return trade_log

def run_mass_parallel_backtest(data_dir, oracle_config):
    files = sorted(glob(f"{data_dir}/*.parquet"))[:300]
    max_workers = max(1, os.cpu_count() - 2)
    tasks = [(f, oracle_config) for f in files]
    all_trades = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(backtest_single_stock, task): task for task in tasks}
        for i, future in enumerate(as_completed(futures)):
            trades = future.result()
            all_trades.extend(trades)
            
    if not all_trades: return
    df = pd.DataFrame(all_trades)
    wins = df[df['pnl_pct'] > 0]['pnl_pct']
    losses = df[df['pnl_pct'] <= 0]['pnl_pct']
    win_rate = len(wins) / len(df)
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 1e-9
    asym_ratio = avg_win / avg_loss
    print(f"Asymmetry Payoff: {asym_ratio:.2f}")

if __name__ == "__main__":
    ORACLE_CONFIG = {
        'model_path': 'omega_2d_oracle_best.pth',
        'days': 15,
        'ticks_per_day': 64,
        'z_score_trigger': -3.0
    }
    run_mass_parallel_backtest("./ticker_matrix_shards_2025_blind", ORACLE_CONFIG)