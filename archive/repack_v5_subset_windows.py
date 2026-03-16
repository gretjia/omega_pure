import polars as pl
import numpy as np
import os
import gc
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# List of high-liquidity symbols
SYMBOLS = ['204001.SZ', '300059.SZ', '601933.SH', '000063.SZ', '001391.SZ', '588000.SZ', '600030.SH', '603019.SH', '510300.SZ', '000977.SZ', '000564.SZ', '002130.SZ', '600839.SH', '002583.SZ', '601727.SH', '002085.SZ', '159915.SZ', '002449.SZ', '000938.SZ', '600255.SH', '002416.SZ', '600171.SH', '600186.SH', '002131.SZ', '000100.SZ', '002156.SZ', '600580.SH', '000725.SZ', '601398.SH', '600050.SH', '002617.SZ', '002717.SZ', '002640.SZ', '600584.SH', '601318.SH', '000681.SZ', '600745.SH', '300383.SZ', '000651.SZ', '512480.SZ', '300607.SZ', '688981.SH', '601138.SH', '601288.SH', '601360.SH', '002361.SZ', '600602.SH', '002241.SZ', '601899.SH', '600398.SH', '300561.SZ', '000810.SZ', '123163.SZ', '002456.SZ', '000066.SZ', '002031.SZ', '000158.SZ', '600246.SH', '001696.SZ', '600973.SH', '300339.SZ', '002475.SZ', '002292.SZ', '002185.SZ', '002195.SZ', '300442.SZ', '002384.SZ', '002230.SZ', '002354.SZ', '113575.SZ', '600104.SH', '601162.SH', '600172.SH', '601456.SH', '603881.SH', '601919.SH', '002797.SZ', '000625.SZ', '600900.SH', '000333.SZ', '002137.SZ', '300017.SZ', '110060.SZ', '601766.SH', '002045.SZ', '002175.SZ', '600157.SH', '600418.SH', '002607.SZ', '002281.SZ', '300548.SZ', '300377.SZ', '162415.SZ', '127106.SZ', '512880.SZ', '588200.SZ', '300323.SZ', '002312.SZ', '603986.SH', '601985.SH']

INPUT_DIR = r"D:\Omega_frames\latest_base_l1\host=linux1"
OUTPUT_DIR = r"D:\ticker_matrix_shards_v5"

def forge_np(price_change, order_flow, market_volume, volatility, dim=10, delay=1, epsilon=0.05):
    safe_volume = np.maximum(market_volume, 1e-8)
    normalized_flow = np.abs(order_flow) / safe_volume
    theoretical_impact = 1.0 * volatility * np.sign(order_flow) * np.sqrt(normalized_flow)
    residuals = price_change - theoretical_impact
    
    valid_length = len(residuals) - (dim - 1) * delay
    if valid_length <= 0:
        return np.stack([price_change, residuals, np.zeros_like(price_change)], axis=-1)
        
    starts = np.arange(valid_length)
    offsets = np.arange(dim) * delay
    manifold = residuals[starts[:, None] + offsets[None, :]]
    
    sq_norms = np.sum(manifold**2, axis=1)
    sq_dists = sq_norms[:, None] + sq_norms[None, :] - 2.0 * np.dot(manifold, manifold.T)
    sq_dists = np.maximum(sq_dists, 0.0)
    connectivity = 1.0 / (1.0 + np.exp(-(epsilon**2 - sq_dists) * 1000.0))
    local_density = np.mean(connectivity, axis=1)
    epiplexity = -np.log(np.maximum(local_density, 1e-8))
    
    pad_len = (dim - 1) * delay
    padded_epiplexity = np.pad(epiplexity, (pad_len, 0), mode='edge')
    return np.stack([price_change, residuals, padded_epiplexity], axis=-1)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.startswith("2025") and f.endswith(".parquet") and f < "20250701"])
    print(f"Processing {len(files)} daily shards for {len(SYMBOLS)} symbols...")

    # Dictionary to hold dataframes per symbol
    symbol_data = {sym: [] for sym in SYMBOLS}

    for f_name in tqdm(files):
        f_path = os.path.join(INPUT_DIR, f_name)
        try:
            # 1. Read and calculate basic features
            df = pl.read_parquet(f_path, columns=["symbol", "price", "vol_tick"])
            df = df.filter(pl.col("symbol").is_in(SYMBOLS)).with_columns([
                pl.col("price").diff().fill_null(0.0).over("symbol").alias("price_change"),
                (pl.col("price").diff().sign() * pl.col("vol_tick")).fill_null(0.0).over("symbol").alias("order_flow"),
                pl.col("price").rolling_std(window_size=64).fill_null(0.001).over("symbol").alias("volatility")
            ])
            
            # 2. Process each symbol
            groups = df.partition_by("symbol", as_dict=True)
            for sym, sym_df in groups.items():
                if len(sym_df) < 128:
                    continue
                
                # Apply forge
                features = forge_np(
                    sym_df["price_change"].to_numpy(),
                    sym_df["order_flow"].to_numpy(),
                    sym_df["vol_tick"].to_numpy(), # market_volume is vol_tick
                    sym_df["volatility"].to_numpy()
                )
                
                # Create final dataframe for this shard/symbol
                final_df = pl.DataFrame({
                    "price_change": features[:, 0],
                    "srl_residual": features[:, 1],
                    "epiplexity": features[:, 2]
                })
                symbol_data[sym].append(final_df)
                
            del df, groups
            gc.collect()
        except Exception as e:
            print(f"Error processing {f_name}: {e}")

    print("Merging and saving ticker shards...")
    for sym, dfs in tqdm(symbol_data.items()):
        if not dfs:
            continue
        final_ticker_df = pl.concat(dfs)
        out_path = os.path.join(OUTPUT_DIR, f"{sym}.parquet")
        final_ticker_df.write_parquet(out_path)
        del final_ticker_df
        gc.collect()

    print("Done.")

if __name__ == "__main__":
    main()
