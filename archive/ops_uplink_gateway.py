"""
THE OMEGA PROTOCOL: UPLINK GATEWAY (The Conveyor Belt)
Execution Target: omega-vm (Master Control)
Action: Safely pull forged shards from workers via HK and stream them to GCP.
Discipline: NEVER hold more than 10GB locally on omega-vm.
"""

import os
import subprocess
import shutil
import time
from pathlib import Path

# =====================================================================
# CONFIGURATION
# =====================================================================
GCS_DESTINATION = "gs://omega-pure-data/base_matrix_shards"
BUFFER_DIR = Path("/tmp/omega_uplink_buffer")

WORKERS = {
    "linux1": {
        "ssh_target": "linux1-lx",
        "remote_dir": "/home/zepher/omega_pure/base_matrix_shards_2024",
        "jump_node": "hk-node",
        "find_cmd": "find /home/zepher/omega_pure/base_matrix_shards_2024 -name '*.parquet' 2>/dev/null"
    },
    "windows1": {
        "ssh_target": "windows1-w1",
        "remote_dir": "C:/omega_pure/base_matrix_shards_2024",
        "jump_node": "hk-node",
        "find_cmd": "powershell -Command \"(Get-ChildItem -Path C:\\omega_pure\\base_matrix_shards_2024 -Filter *.parquet).FullName\" 2> $null"
    }
}

def pull_and_upload(name, config):
    print(f"\n[UPLINK] Initiating radar sweep on {name}...")
    
    target = config["ssh_target"]
    jump = config["jump_node"]
    
    # 1. Get list of files
    try:
        res = subprocess.run(
            ["ssh", "-J", jump, target, config["find_cmd"]],
            capture_output=True, text=True, check=True
        )
        files = [f.strip() for f in res.stdout.splitlines() if f.strip() and f.strip().endswith(".parquet")]
    except subprocess.CalledProcessError:
        print(f"  [WARN] No shards found or directory does not exist on {name}.")
        return

    if not files:
        print(f"  [INFO] No completed shards waiting on {name}.")
        return

    print(f"  [INFO] Detected {len(files)} forged shards. Initiating secure transport via {jump}...")
    
    # Process in small chunks to prevent omega-vm disk explosion and SSH timeout
    CHUNK_SIZE = 10
    for i in range(0, len(files), CHUNK_SIZE):
        chunk = files[i:i + CHUNK_SIZE]
        BUFFER_DIR.mkdir(parents=True, exist_ok=True)
        
        print(f"  [TRANSPORT] Pulling chunk {i//CHUNK_SIZE + 1} ({len(chunk)} files)...", flush=True)
        
        for file in chunk:
            # Handle Windows path formatting output correctly
            if "\\" in file:
                 remote_file = file.replace("\\", "/") # Quick fix for SCP path format
            else:
                 remote_file = file
                 
            # Note: Removed the extra quotes around remote_file which caused SCP to look for literally '"path"'
            scp_cmd = [
                "scp", "-o", f"ProxyJump={jump}",
                f"{target}:{remote_file}",
                str(BUFFER_DIR)
            ]
            try:
                subprocess.run(scp_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"    -> Pulled: {os.path.basename(remote_file)}", flush=True)
            except subprocess.CalledProcessError as e:
                print(f"    -> Failed to pull {os.path.basename(remote_file)}. It might have been deleted or network dropped.", flush=True)
            
        print(f"  [GCP UPLOAD] Beam chunk to {GCS_DESTINATION}...")
        try:
             subprocess.run(["gcloud", "storage", "cp", f"{BUFFER_DIR}/*.parquet", f"{GCS_DESTINATION}/"], check=True, stdout=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
             print(f"  [FATAL] GCS Upload failed! Halting conveyor belt. Error: {e}")
             return

        print(f"  [CLEANUP] Incinerating local buffer on omega-vm...")
        shutil.rmtree(BUFFER_DIR)
        
        # Optional: We could also delete the files on the worker to free up their space, 
        # but in Talebian operations, we preserve the local physical state until fully verified.
        
    print(f"✔️ [UPLINK SUCCESS] {name} payload completely delivered to GCP.")

if __name__ == "__main__":
    print("="*60)
    print("🛸 OMEGA PROTOCOL: THE CONVEYOR BELT")
    print("="*60)
    for name, config in WORKERS.items():
        pull_and_upload(name, config)
