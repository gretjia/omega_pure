"""
THE OMEGA PROTOCOL: END-TO-END SMOKE TEST
Execution Target: Any machine with CPU (Mac/Omega-VM/Linux)
Purpose: Validate structural integrity from Step 1 (Raw) to Step 4 (Backtest).
"""

import os
import shutil
import torch
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path

# Imbued from Omega Pure steps
from omega_epiplexity_forge import forge_epiplexity_tensor
from vertex_mae_compressor import EpiplexityMAE, EpiplexityShardDataset
from omega_crucible import TalebianOracle, stream_blind_backtest

def main():
    print("="*60)
    print("🚀 OMEGA PROTOCOL: END-TO-END SMOKE TEST INITIATED")
    print("="*60)
    
    TEST_DIR = Path("./smoke_test_env")
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
    TEST_DIR.mkdir()

    try:
        # =====================================================================
        # 1. MOCKING THE L1 BASE LAKE (Simulating Old Project Output)
        # =====================================================================
        print("\n[STEP 0] Mocking 10,000 ticks of Base_L1 Parquet...")
        np.random.seed(42)
        n_ticks = 10000
        
        # Simulating random walk with occasional "institutional" low-entropy blasts
        prices = np.cumsum(np.random.randn(n_ticks) * 0.01) + 100.0
        vols = np.abs(np.random.randn(n_ticks) * 1000)
        
        # Inject an anomaly at tick 5000 (The Kill Switch Target)
        prices[5000:5064] += np.linspace(0, 0.5, 64) # Directional push
        vols[5000:5064] *= 10.0 # Huge volume
        
        df_mock = pl.DataFrame({
            "symbol": ["000001.SZ"] * n_ticks,
            "time": np.arange(n_ticks),
            "price": prices,
            "vol_tick": vols
        })
        
        # =====================================================================
        # 2. STEP 1 & 2: THE TENSOR MATERIALIZER & JAX FORGE
        # =====================================================================
        print("\n[STEP 1 & 2] Extracting physical columns & JAX Epiplexity Forging...")
        
        # Polars highly-optimized extraction (What omega_tensor_materializer.py will do)
        df_features = df_mock.with_columns([
            pl.col("price").diff().fill_null(0.0).alias("price_change"),
            (pl.col("price").diff().sign() * pl.col("vol_tick")).fill_null(0.0).alias("order_flow"),
            pl.col("price").rolling_std(window_size=64).fill_null(0.001).alias("volatility")
        ])
        
        # Convert to arrays for JAX
        arr_price_change = df_features["price_change"].to_numpy().astype(np.float32)
        arr_order_flow = df_features["order_flow"].to_numpy().astype(np.float32)
        arr_market_volume = df_features["vol_tick"].to_numpy().astype(np.float32)
        arr_volatility = df_features["volatility"].to_numpy().astype(np.float32)
        
        # ⚠️ THE JAX KERNEL EXECUTION ⚠️
        import jax # Ensures JAX doesn't crash on CPU
        jax.config.update("jax_platform_name", "cpu") # Force CPU for smoke test
        
        feature_tensor = forge_epiplexity_tensor(
            price_change=arr_price_change,
            order_flow=arr_order_flow,
            market_volume=arr_market_volume,
            volatility=arr_volatility,
            dim=10, delay=1
        )
        
        # feature_tensor output shape: (N, 3) -> [price_change, srl_residual, epiplexity]
        shard_path = TEST_DIR / "smoke_2025_01.parquet"
        
        pd.DataFrame({
            "price": df_features["price"].to_numpy(),
            "order_flow": arr_order_flow,
            "price_change": np.asarray(feature_tensor[:, 0]),
            "srl_residual": np.asarray(feature_tensor[:, 1]),
            "epiplexity": np.asarray(feature_tensor[:, 2])
        }).to_parquet(shard_path)
        print(f"✔️ Step 1 & 2 Complete. Shard materialized at {shard_path}. Shape: {feature_tensor.shape}")

        # =====================================================================
        # 3. STEP 3: MAE TRAINING (Simulating Vertex AI)
        # =====================================================================
        print("\n[STEP 3] Igniting Masked Autoencoder (Mock GCP Training)...")
        # We run 1 epoch just to test forward/backward pass and saving
        dataset = EpiplexityShardDataset(str(TEST_DIR), seq_len=64)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=256)
        
        model = EpiplexityMAE(seq_len=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        model.train()
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            loss, _ = model(batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        
        model_path = TEST_DIR / "smoke_mae_oracle.pth"
        torch.save({
            'state_dict': model.state_dict(),
            'mean': dataset.mean,
            'std': dataset.std,
            'baseline_loss': avg_loss
        }, model_path)
        print(f"✔️ Step 3 Complete. Model saved. Baseline Loss: {avg_loss:.6f}")
        
        # =====================================================================
        # 4. STEP 4: THE TALEBIAN ORACLE (Zero-Copy Backtest)
        # =====================================================================
        print("\n[STEP 4] Awakening the Talebian Oracle for Blind Backtest...")
        
        # Because our random data is mostly noise, we set a high threshold just to trigger a trade in the test
        oracle = TalebianOracle(model_path=str(model_path), z_score_threshold=0.0) 
        
        stream_blind_backtest(data_dir=str(TEST_DIR), oracle=oracle, holding_ticks=100)
        
        print("\n" + "="*60)
        print("🟢 SMOKE TEST PASSED: Full Data Pipeline is Structurally Sound.")
        print("="*60)
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"🔴 SMOKE TEST FAILED: Downstream breakage detected!")
        print(f"Error: {str(e)}")
        print("="*60)
        raise e
    finally:
        # Cleanup
        shutil.rmtree(TEST_DIR, ignore_errors=True)

if __name__ == "__main__":
    main()