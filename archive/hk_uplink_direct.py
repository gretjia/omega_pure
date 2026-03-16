import subprocess
import os
import time
from concurrent.futures import ThreadPoolExecutor

def upload_file(args):
    file, target, gcs_bucket, tmp_dir = args
    basename = os.path.basename(file)
    local_path = os.path.join(tmp_dir, basename)
    
    # Check if exists on GCS (Run directly on HK node)
    res = subprocess.run(["/home/ubuntu/google-cloud-sdk/bin/gcloud", "storage", "ls", f"{gcs_bucket}/{basename}"], capture_output=True)
    if res.returncode == 0:
        return f"⏩ [SKIP] {basename} already in GCS"
        
    # Pull via SCP directly from linux1-lx (no jump node needed since they are both Tailscale/Public accessible via the key)
    # HK node has a fast 200M connection, so this pull should be extremely fast.
    scp_cmd = ["scp", "-q", "-i", os.path.expanduser("~/.ssh/id_ed25519_omega_workers"), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=30", f"{target}:{file}", local_path]
    try:
        subprocess.run(scp_cmd, check=True)
    except Exception as e:
        return f"❌ [PULL FAIL] {basename}"
        
    # Upload to GCS directly from HK
    try:
        subprocess.run(["/home/ubuntu/google-cloud-sdk/bin/gcloud", "storage", "cp", local_path, f"{gcs_bucket}/"], check=True, capture_output=True)
        os.remove(local_path)
        return f"✔️ [DONE] {basename}"
    except Exception as e:
        if os.path.exists(local_path): os.remove(local_path)
        return f"❌ [UPLOAD FAIL] {basename}"

if __name__ == "__main__":
    target = "zepher@100.64.97.113"
    remote_dir = "/home/zepher/omega_pure/base_matrix_shards_2024"
    gcs_bucket = "gs://omega-pure-data/base_matrix_shards"
    tmp_dir = "/tmp/omega_chunk"
    
    os.makedirs(tmp_dir, exist_ok=True)
    
    print("📡 Fetching list of files directly from Linux1...", flush=True)
    res = subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no", "-i", os.path.expanduser("~/.ssh/id_ed25519_omega_workers"), target, f"ls -1 {remote_dir}/*.parquet"], capture_output=True, text=True)
    files = [f.strip() for f in res.stdout.splitlines() if f.strip().endswith('.parquet')]
    
    print(f"🎯 Found {len(files)} shards. Starting hyper-speed transport with 8 threads...", flush=True)
    
    args_list = [(f, target, gcs_bucket, tmp_dir) for f in files]
    
    # We can use 8-10 threads here because it's a direct connection without a jump node bottleneck
    with ThreadPoolExecutor(max_workers=8) as executor:
        for i, result in enumerate(executor.map(upload_file, args_list)):
            print(f"[{i+1}/{len(files)}] {result}", flush=True)