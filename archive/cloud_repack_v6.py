import polars as pl
import os
import subprocess
import gc
from tqdm import tqdm
import gcsfs
from collections import defaultdict

BUCKET = "gs://omega-pure-data"
INPUT_DIR = "omega-pure-data/base_matrix_shards"
OUTPUT_PATH = f"{BUCKET}/ticker_matrix_shards"

def main():
    fs = gcsfs.GCSFileSystem()
    print(f"Scanning input files: {INPUT_DIR}/2025*.parquet")
    all_files = sorted(fs.glob(f"{INPUT_DIR}/2025*.parquet"))
    
    if not all_files:
        print("No files found!")
        return

    # 1. Group files by Ticker Hash (suffix)
    # Filename format: path/to/20250102_fbd5c8b.parquet
    ticker_groups = defaultdict(list)
    for f in all_files:
        name = f.split("/")[-1]
        ticker_hash = name.split("_")[-1].replace(".parquet", "")
        ticker_groups[ticker_hash].append(f)
    
    print(f"Found {len(ticker_groups)} unique tickers.")
    
    # 2. Pick top 100 tickers with most shards (most history)
    sorted_tickers = sorted(ticker_groups.items(), key=lambda x: len(x[1]), reverse=True)
    top_tickers = sorted_tickers[:100]
    
    print(f"Processing Top 100 tickers...")
    
    os.makedirs("/tmp/repack", exist_ok=True)
    
    for ticker_hash, files in tqdm(top_tickers):
        # Sort files by date (first part of filename)
        files = sorted(files) 
        
        # Read and concat all shards for this ticker
        df_list = []
        for f in files:
            # Note: We use gs:// prefix for polars read
            try:
                # Read without 'symbol' column
                df = pl.read_parquet(f"gs://{f}", columns=["price_change", "srl_residual", "epiplexity"])
                df_list.append(df)
            except Exception as e:
                print(f"Error reading {f}: {e}")
        
        if not df_list:
            continue
            
        ticker_df = pl.concat(df_list)
        
        # Write to local temp
        local_file = f"/tmp/repack/{ticker_hash}.parquet"
        ticker_df.write_parquet(local_file)
        
        # Upload to GCS
        remote_file = f"{OUTPUT_PATH}/{ticker_hash}.parquet"
        subprocess.run(["gcloud", "storage", "cp", local_file, remote_file], check=True, capture_output=True)
        
        # Cleanup
        os.remove(local_file)
        del ticker_df, df_list
        gc.collect()

    print("Repacking complete.")

if __name__ == "__main__":
    main()
