"""
THE OMEGA PROTOCOL: WORKER DEPLOYMENT
Execution Target: omega-vm (Master Control)
Action: Synchronize pure code to workers via secure HK jump node. NO GIT ON WORKERS.
"""

import os
import subprocess
from pathlib import Path

# =====================================================================
# HARDWARE TOPOLOGY TARGETS
# =====================================================================
WORKERS = {
    "linux1": {
        "ssh_target": "linux1-lx",
        "remote_dir": "/home/zepher/omega_pure",
        "jump_node": "hk-node"
    },
    "windows1": {
        "ssh_target": "windows1-w1",
        "remote_dir": "C:/omega_pure",
        "jump_node": "hk-node",
        "is_cygwin": True # Assuming Windows uses cygwin/mingw rsync or native ssh
    }
}

LOCAL_DIR = Path(__file__).parent.resolve()

def deploy_to_worker(name, config):
    print(f"\n[DEPLOY] Establishing secure link to {name} via {config['jump_node']}...")
    target = config["ssh_target"]
    remote_dir = config["remote_dir"]
    
    # Create remote directory
    mkdir_cmd = ["ssh", "-J", config["jump_node"], target, f"mkdir -p {remote_dir}"]
    if config.get("is_cygwin"):
         mkdir_cmd = ["ssh", "-J", config["jump_node"], target, f"powershell -Command \"New-Item -ItemType Directory -Force -Path {remote_dir}\""]
         
    try:
        subprocess.run(mkdir_cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"  [WARN] Failed to create directory on {name}: {e}")

    # Rsync only python scripts to keep it PURE. Exclude git, venv, caches.
    print(f"  [SYNC] Pushing Python tensor matrices to {target}:{remote_dir}")
    
    # Use standard SCP for pure cross-platform reliability instead of rsync, 
    # since Windows rsync paths (like /cygdrive/c/) can be messy to map automatically.
    py_files = list(LOCAL_DIR.glob("*.py"))
    
    for py_file in py_files:
        scp_cmd = [
            "scp",
            "-o", f"ProxyJump={config['jump_node']}",
            str(py_file),
            f"{target}:{remote_dir}/{py_file.name}"
        ]
        try:
             subprocess.run(scp_cmd, check=True, stdout=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
             print(f"  [ERROR] Failed to push {py_file.name} to {name}.")

    print(f"✔️ [DEPLOY SUCCESS] {name} is armed and ready.")

if __name__ == "__main__":
    print("="*60)
    print("🚀 OMEGA PROTOCOL: ORCHESTRATION SYNC")
    print("="*60)
    for name, config in WORKERS.items():
        deploy_to_worker(name, config)
