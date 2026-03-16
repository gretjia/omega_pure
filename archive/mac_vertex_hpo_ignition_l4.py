"""
THE OMEGA PROTOCOL: VERTEX IGNITION (THE SCIENTIFIC DISCOVERY)
Target: Google Cloud Vizier (Bayesian Optimization)
Fleet: L4 Wolfpack (Single-GPU instances with massive RAM to bypass Queues & OOM)
"""
from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt

# =====================================================================
PROJECT_ID = "gen-lang-client-0250995579"           # <-- 必须替换
REGION = "us-central1"                       
BUCKET_URI = "gs://omega-pure-data"        # <-- 必须替换

GCS_INPUT = f"{BUCKET_URI}/base_matrix_shards" 
GCS_OUTPUT = f"{BUCKET_URI}/omega_models/hpo_trials_l4_wolfpack"
# =====================================================================

def launch_scientific_hpo():
    print("[MAC ORACLE] Uploading Compressor Payload to GCS...")
    import subprocess
    subprocess.run(["gcloud", "storage", "cp", "vertex_mae_compressor_l4.py", f"{BUCKET_URI}/scripts/vertex_mae_compressor_l4.py"], check=True)

    print("[MAC ORACLE] Authenticating with Vertex AI Vizier...")
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)

    # 🚀 L4 狼群配置：g2-standard-32 拥有 32核CPU 和 128GB 的物理内存！
    # 这道物理铁壁将彻底粉碎 Linux OOM Killer。
    worker_pool_specs = [{
        "machine_spec": {
            "machine_type": "g2-standard-32",
            "accelerator_type": "NVIDIA_L4",
            "accelerator_count": 1,  # 斩断 DataParallel，单机单卡
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1.py310:latest",
            "command": ["bash", "-c"],
            # 🚀 [架构师防漏盾] "_" 占位符极其关键，防止 Bash 吞噬 Vizier 注入的第一个超参
            "args": [
                f"gcloud storage cp {BUCKET_URI}/scripts/vertex_mae_compressor_l4.py . && pip install gcsfs pandas pyarrow numpy cloudml-hypertune && python3 vertex_mae_compressor_l4.py --gcs_input={GCS_INPUT} --epochs=10 --batch_size=4096 \"$@\"",
                "_"  
            ]
        }
    }]

    custom_job = aiplatform.CustomJob(
        display_name="omega-mae-hpo-worker-l4",
        worker_pool_specs=worker_pool_specs,
        base_output_dir=GCS_OUTPUT
    )

    parameter_spec = {
        "stride": hpt.IntegerParameterSpec(min=100, max=1500, scale="linear"), 
        "seq_len": hpt.DiscreteParameterSpec(values=[32, 64, 128], scale=None),
        "lookback_days": hpt.IntegerParameterSpec(min=3, max=30, scale="linear"),
    }

    metric_spec = {"val_fvu": "minimize"}

    print("[MAC ORACLE] Unleashing L4 Wolfpack. Firing HPO Payload...")
    hpt_job = aiplatform.HyperparameterTuningJob(
        display_name="omega-macro-discovery-l4",
        custom_job=custom_job,
        metric_spec=metric_spec,
        parameter_spec=parameter_spec,
        max_trial_count=30,       # 虫群总试验次数
        parallel_trial_count=20,   # 🔥 并发拉起 20 台 L4 机器同时跑！告别排队！(占用 20/100 额度，安全起见先不拉满100防止突发计费异常)
        search_algorithm=None,    
    )

    hpt_job.run(sync=False)
    print(f"[MAC ORACLE] 🚀 Scientific Search initiated on L4 GPUs.")
    print(f"[MAC ORACLE] The Deadlock and OOM Traps have been neutralized. Monitor GCP Console.")

if __name__ == "__main__":
    launch_scientific_hpo()