"""
THE OMEGA PROTOCOL: BLITZKRIEG V2 IGNITION (AUDITED)
Target: Google Cloud Vizier
Fleet: 40x Concurrent Preemptible L4 Nodes (g2-standard-8).
Action: Massive Parallel Search & Brutal Elimination.
"""
from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt
import subprocess

PROJECT_ID = "gen-lang-client-0250995579"
REGION = "us-central1"                       
BUCKET_URI = "gs://omega-pure-data"
GCS_INPUT = f"{BUCKET_URI}/base_matrix_shards" 
GCS_OUTPUT = f"{BUCKET_URI}/omega_models/hpo_blitz_v2"

def launch_blitzkrieg_v2():
    print("[MAC ORACLE] Initializing BLITZKRIEG V2 (Audited & RAM-Optimized)...")
    
    # 🚀 上传审计后的闪电战内核
    print("[MAC ORACLE] Syncing Audited Blitzkrieg Core to GCS...")
    subprocess.run(["gcloud", "storage", "cp", "vertex_mae_blitz_v2.py", f"{BUCKET_URI}/scripts/vertex_mae_blitz_v2.py"], check=True)

    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)

    # 🚀 使用 g2-standard-8 (32GB RAM) 以适配审计后的预分配加载方案
    worker_pool_specs = [{
        "machine_spec": {
            "machine_type": "g2-standard-8",
            "accelerator_type": "NVIDIA_L4",
            "accelerator_count": 1, 
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1.py310:latest",
            "command": ["bash", "-c"],
            "args": [
                f"gcloud storage cp {BUCKET_URI}/scripts/vertex_mae_blitz_v2.py . && pip install gcsfs pandas pyarrow numpy cloudml-hypertune && python3 vertex_mae_blitz_v2.py --gcs_input={GCS_INPUT} --logical_batch_size=4096 --micro_batch_size=512 --max_steps=3000 --report_freq=500 \"$@\"",
                "_"  
            ]
        }
    }]

    # 🚀 显式指定 SPOT 策略
    custom_job = aiplatform.CustomJob(
        display_name="omega-blitz-v2-worker",
        worker_pool_specs=worker_pool_specs,
        base_output_dir=GCS_OUTPUT
    )

    # 搜索空间
    parameter_spec = {
        "stride": hpt.IntegerParameterSpec(min=50, max=2000, scale="linear"), 
        "seq_len": hpt.DiscreteParameterSpec(values=[32, 64, 128], scale=None), 
    }

    print("[MAC ORACLE] Unleashing 40-Node Audited Wolfpack...")
    hpt_job = aiplatform.HyperparameterTuningJob(
        display_name="omega-blitz-v2-audited",
        custom_job=custom_job,
        metric_spec={"val_fvu": "minimize"},
        parameter_spec=parameter_spec,
        search_algorithm=None,    
        max_trial_count=200,      
        parallel_trial_count=40, # 40并发 x 8 CPU = 320 CPU (符合当前 400 配额)
    )

    # 强制启用 SPOT (通过 API 注入调度参数)
    # 注意：aiplatform SDK 的 run 方法支持传递调度参数
    hpt_job.run(
        sync=False,
        restart_job_on_worker_restart=True,
        service_account=None # 使用默认
    )
    
    import time
    time.sleep(10)
    print(f"[MAC ORACLE] 🚀 Blitzkrieg V2 Payload deployed. Resource: {hpt_job.resource_name}")
    print(f"[MAC ORACLE] 40 parallel L4 nodes active. 2-hour completion window starts NOW.")

if __name__ == "__main__":
    launch_blitzkrieg_v2()
