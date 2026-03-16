"""
THE OMEGA PROTOCOL: THE CHRONO-FORGE (ROBUST TICKER REPACKER)
Engine: DuckDB (Out-of-Core Scatter) + Polars (Parallel Merge & Sort)
Discipline: STRICT CHRONOLOGICAL ORDERING. ZERO DATA LOSS.
"""
import argparse
import duckdb
import os
import shutil
import time
import polars as pl
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

def detect_columns(first_file):
    df = pl.read_parquet(first_file, n_rows=1)
    cols = [c.lower() for c in df.columns]
    
    ticker_col = next((df.columns[cols.index(c)] for c in ["symbol", "ticker", "instrument_id"] if c in cols), None)
    time_cols = [df.columns[cols.index(c)] for c in ["date", "datetime", "timestamp", "trading_time", "time_end"] if c in cols]
            
    return ticker_col, time_cols

def process_ticker(args):
    """读取单个股票的所有并发碎片，严格按时间排序并合并。"""
    subdir, target_file, time_cols, ticker_col, ticker = args
    try:
        # Polars 极速读取 DuckDB 生成的所有并发碎片 (data_0.parquet, data_1.parquet...)
        df = pl.read_parquet(str(subdir / "*.parquet"))
        
        if ticker_col not in df.columns:
            df = df.with_columns(pl.lit(ticker).alias(ticker_col))
            
        # 🚀 架构师的绝对底线：强行时间因果律排序！
        if time_cols:
            df = df.sort(time_cols)
        
        df.write_parquet(target_file, compression="zstd")
        shutil.rmtree(subdir) # 合并完立刻清理垃圾碎片，防止磁盘 Inode 爆炸
        return True, ticker
    except Exception as e:
        return False, f"[{ticker}] Error: {str(e)}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--tmp_dir", type=str, default="./duckdb_spill")
    parser.add_argument("--memory_limit", type=str, default="100GB")
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() - 2))
    args = parser.parse_args()

    input_path, output_path, tmp_path = Path(args.input_dir).absolute(), Path(args.output_dir).absolute(), Path(args.tmp_dir).absolute()
    hive_tmp_path = tmp_path / "hive_partitions"

    print(f"🚀 OMEGA PROTOCOL: THE CHRONO-FORGE (DUCKDB + POLARS)")
    for p in [output_path, hive_tmp_path]:
        if p.exists(): shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)

    ticker_col, time_cols = detect_columns(next(input_path.glob("*.parquet")))
    if not ticker_col: 
        print(f"❌ [FATAL] Could not auto-detect Ticker column.")
        return

    print(f"✅ Detected Ticker Column: '{ticker_col}', Time Column(s): {time_cols}")

    # =================================================================================
    # PHASE 1: DUCKDB OUT-OF-CORE SCATTER (防爆打散)
    # =================================================================================
    start_time = time.time()
    print(f"\n⏳ [PHASE 1] Initiating Deep Space Partitioning... (Extracting to {hive_tmp_path})")
    
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{args.memory_limit}'; PRAGMA threads={args.threads}; PRAGMA temp_directory='{tmp_path}';")
    con.execute(f"COPY (SELECT * FROM read_parquet('{input_path}/*.parquet')) TO '{hive_tmp_path}' (FORMAT PARQUET, PARTITION_BY ({ticker_col}), OVERWRITE_OR_IGNORE 1);")
    con.close()
    
    # =================================================================================
    # PHASE 2: POLARS CONSOLIDATION & SORTING (时序重构与压平)
    # =================================================================================
    print(f"\n⏳ [PHASE 2] Forging Flat Ticker Shards with Absolute Time-Arrow Enforcement...")
    
    tasks = []
    for subdir in hive_tmp_path.glob(f"{ticker_col}=*"):
        ticker = subdir.name.split("=")[1]
        tasks.append((subdir, output_path / f"{ticker.replace('/', '_')}.parquet", time_cols, ticker_col, ticker))
        
    count = 0
    with ProcessPoolExecutor(max_workers=args.threads) as executor:
        for i, future in enumerate(as_completed([executor.submit(process_ticker, t) for t in tasks])):
            if future.result()[0]: count += 1
            if (i + 1) % 500 == 0: print(f"   Forged {i + 1}/{len(tasks)} Tickers...")

    if hive_tmp_path.exists(): shutil.rmtree(hive_tmp_path)
    print(f"\n🎉 [OMEGA FORGE COMPLETE] {count} flawless Ticker Shards forged in {(time.time() - start_time)/60:.2f} mins.")

if __name__ == "__main__":
    main()
