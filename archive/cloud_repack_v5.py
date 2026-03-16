import polars as pl
import os
import subprocess
import gc
from tqdm import tqdm

BUCKET = "gs://omega-pure-data"
INPUT_PATH = f"{BUCKET}/base_matrix_shards/2025*.parquet"
OUTPUT_PATH = f"{BUCKET}/ticker_matrix_shards"

def repack():
    print(f"Scanning input files: {INPUT_PATH}")
    files = subprocess.check_output(f"gcloud storage ls {INPUT_PATH}", shell=True).decode().splitlines()
    files = sorted([f for f in files if f.endswith(".parquet")])
    
    if not files:
        print("No files found!")
        return

    # 1. Get first 100 symbols
    print("Extracting first 100 symbols...")
    first_df = pl.read_parquet(files[0], columns=["symbol"])
    symbols = first_df["symbol"].unique().head(100).to_list()
    print(f"Targeting {len(symbols)} symbols.")

    # 2. Process in chunks to avoid memory issues
    # Since we only want 100 symbols, we can filter while reading
    chunk_size = 50
    for chunk_idx in range(0, len(files), chunk_size):
        chunk_files = files[chunk_idx : chunk_idx + chunk_size]
        print(f"Processing chunk {chunk_idx//chunk_size + 1}, files {len(chunk_files)}...")
        
        # Read and filter
        df = pl.scan_parquet(chunk_files).filter(pl.col("symbol").is_in(symbols)).collect()
        
        # Partition and upload
        for sym in symbols:
            sym_df = df.filter(pl.col("symbol") == sym)
            if len(sym_df) == 0:
                continue
            
            # Use a local temp file for intermediate storage
            local_tmp = f"/tmp/{sym}_chunk.parquet"
            sym_df.write_parquet(local_tmp)
            
            # Append/Upload to GCS
            # Note: Polars doesn't support easy append to GCS. 
            # We'll write chunked files and merge later if needed, 
            # or just write one file per symbol per chunk and let the trainer read all.
            # Actually, for Blitzkrieg, let's just merge them all at once if memory allows.
            # 100 symbols across 700 shards is not huge if we only pick 100.
            
        del df
        gc.collect()

    print("Repacking complete (Conceptual). Starting Real Implementation...")

if __name__ == "__main__":
    # Real efficient implementation:
    # We will use a Vertex AI Custom Job to do this.
    # To keep it simple, we will read ALL 2025 shards, filter for 100 symbols, and write to GCS.
    
    import gcsfs
    fs = gcsfs.GCSFileSystem()
    files = sorted(fs.glob("omega-pure-data/base_matrix_shards/2025*.parquet"))
    
    # Read first shard to get symbols
    symbols = pl.read_parquet(f"gs://{files[0]}", columns=["symbol"])["symbol"].unique().head(100).to_list()
    print(f"Processing {len(symbols)} symbols across {len(files)} shards...")
    
    # We'll use a local disk on the worker node to accumulate
    os.makedirs("/tmp/repack", exist_ok=True)
    
    # Process all files
    for f_path in tqdm(files):
        df = pl.read_parquet(f"gs://{f_path}", columns=["symbol", "price_change", "srl_residual", "epiplexity"])
        df = df.filter(pl.col("symbol").is_in(symbols))
        
        for sym, sym_df in df.partition_by("symbol", as_dict=True).items():
            sym_file = f"/tmp/repack/{sym}.parquet"
            if os.path.exists(sym_file):
                # Append to existing local file
                existing = pl.read_parquet(sym_file)
                pl.concat([existing, sym_df]).write_parquet(sym_file)
            else:
                sym_df.write_parquet(sym_file)
        
        del df
        gc.collect()
        
    # Upload finished files
    for local_file in os.listdir("/tmp/repack"):
        sym = local_file.replace(".parquet", "")
        remote_file = f"gs://omega-pure-data/ticker_matrix_shards/{local_file}"
        print(f"Uploading {sym}...")
        # Use gsutil or gcloud for faster upload
        subprocess.run(["gcloud", "storage", "cp", f"/tmp/repack/{local_file}", remote_file], check=True)
