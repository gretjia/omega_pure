import subprocess
import os
import time
from concurrent.futures import ThreadPoolExecutor

def upload_file(args):
    file, jump, target, remote_dir, gcs_bucket, tmp_dir = args
    basename = os.path.basename(file)
    local_path = os.path.join(tmp_dir, basename)
    
    # Check if exists on GCS
    res = subprocess.run(["gcloud", "storage", "ls", f"{gcs_bucket}/{basename}"], capture_output=True)
    if res.returncode == 0:
        return f"⏩ [SKIP] {basename} already in GCS"
        
    # Retry loop for SCP since hk-node drops connections randomly
    max_retries = 3
    pulled = False
    
    for attempt in range(max_retries):
        if os.path.exists(local_path):
            os.remove(local_path) # Clean partial downloads before retry
            
        scp_cmd = ["scp", "-q", "-o", f"ProxyJump={jump}", "-o", "ConnectTimeout=30", "-o", "ServerAliveInterval=15", f"{target}:{file}", local_path]
        try:
            subprocess.run(scp_cmd, check=True)
            pulled = True
            break # Success
        except Exception as e:
            time.sleep(2) # Give network a breath
            
    if not pulled:
        return f"❌ [PULL FAIL] {basename} after {max_retries} retries"
        
    # Upload to GCS
    try:
        subprocess.run(["gcloud", "storage", "cp", local_path, f"{gcs_bucket}/"], check=True, capture_output=True)
        os.remove(local_path)
        return f"✔️ [DONE] {basename}"
    except Exception as e:
        if os.path.exists(local_path): os.remove(local_path)
        return f"❌ [UPLOAD FAIL] {basename}"

if __name__ == "__main__":
    jump = "hk-node"
    target = "linux1-lx"
    remote_dir = "/home/zepher/omega_pure/base_matrix_shards_2024"
    gcs_bucket = "gs://omega-pure-data/base_matrix_shards"
    tmp_dir = "/tmp/omega_chunk"
    
    os.makedirs(tmp_dir, exist_ok=True)
    
    print("📡 Fetching list of files...", flush=True)
    res = subprocess.run(["ssh", "-J", jump, target, f"ls -1 {remote_dir}/*.parquet"], capture_output=True, text=True)
    files = [f.strip() for f in res.stdout.splitlines() if f.strip().endswith('.parquet')]
    
    import random
    random.shuffle(files)
    
    # We will use 2 threads. The network drops connections if we open too many tunnels simultaneously through the jump node.
    workers = 2
    print(f"🎯 Found {len(files)} shards. Starting highly resilient transport with {workers} threads...", flush=True)
    
    args_list = [(f, jump, target, remote_dir, gcs_bucket, tmp_dir) for f in files]
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for i, result in enumerate(executor.map(upload_file, args_list)):
            print(f"[{i+1}/{len(files)}] {result}", flush=True)