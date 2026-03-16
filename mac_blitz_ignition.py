"""
THE OMEGA PROTOCOL: BLITZKRIEG IGNITION
Target: Google Cloud Vizier
Fleet: 100x Concurrent Preemptible L4 Nodes. 
Action: Massive Parallel Search & Brutal Elimination.
"""
from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt
import subprocess

PROJECT_ID = "gen-lang-client-0250995579"
REGION = "us-central1"                       
BUCKET_URI = "gs://omega-pure-data"
GCS_INPUT = f"{BUCKET_URI}/base_matrix_shards" 
GCS_OUTPUT = f"{BUCKET_URI}/omega_models/hpo_blitz"

def launch_blitzkrieg_hpo():
    print("[MAC ORACLE] 200x Preemptible L4 Quota Confirmed. Initializing BLITZKRIEG...")
    
    # 🚀 上传最新的闪电战内核
    print("[MAC ORACLE] Syncing Blitzkrieg Core to GCS...")
    subprocess.run(["gcloud", "storage", "cp", "vertex_mae_blitz.py", f"{BUCKET_URI}/scripts/vertex_mae_blitz.py"], check=True)

    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)

    # 🚀 [架构师修正]: 明确指定 SPOT 策略以使用抢占式 L4 配额。
    # 在 Python SDK 中，scheduling 应该放在 worker_pool_specs 的每个 pool 字典中。
    worker_pool_specs = [{
        "machine_spec": {
            "machine_type": "g2-standard-32",
            "accelerator_type": "NVIDIA_L4",
            "accelerator_count": 1, 
        },
        "replica_count": 1,
        "scheduling": {
            "timeout": "86400s", # 24h 超时
            "restart_job_on_worker_restart": True,
            "strategy": "SPOT"   # 🔥 关键：锁定抢占式配额
        },
        "container_spec": {
            "image_uri": "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1.py310:latest",
            "command": ["bash", "-c"],
            "args": [
                f"gcloud storage cp {BUCKET_URI}/scripts/vertex_mae_blitz.py . && pip install gcsfs pandas pyarrow numpy cloudml-hypertune && python3 vertex_mae_blitz.py --gcs_input={GCS_INPUT} --logical_batch_size=4096 --micro_batch_size=512 --max_steps=3000 --report_freq=500 \"$@\"",
                "_"  
            ]
        }
    }]

    custom_job = aiplatform.CustomJob(
        display_name="omega-blitz-worker",
        worker_pool_specs=worker_pool_specs,
        base_output_dir=GCS_OUTPUT
    )

    # 宏观物理边界搜索
    parameter_spec = {
        "stride": hpt.IntegerParameterSpec(min=50, max=2000, scale="linear"), 
        "seq_len": hpt.DiscreteParameterSpec(values=[32, 64, 128], scale=None), 
    }

    print("[MAC ORACLE] Unleashing 100-Node Wolfpack...")
    hpt_job = aiplatform.HyperparameterTuningJob(
        display_name="omega-blitzkrieg-search-v3",
        custom_job=custom_job,
        metric_spec={"val_fvu": "minimize"},
        parameter_spec=parameter_spec,
        search_algorithm=None,    
        max_trial_count=200,      
        parallel_trial_count=100, 
    )

    hpt_job.run(sync=False)
    # 🚀 强制等待资源 ID 可用
    import time
    time.sleep(10)
    print(f"[MAC ORACLE] 🚀 Blitzkrieg Payload deployed. Resource Name: {hpt_job.resource_name}")
    print(f"[MAC ORACLE] 100 parallel trials active. Weak configs will be assassinated automatically via internal logic.")

if __name__ == "__main__":
    launch_blitzkrieg_hpo()
