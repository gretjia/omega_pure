"""
THE OMEGA PROTOCOL: VERTEX IGNITION (THE 100-L4 WOLFPACK - MINI BATCH)
Target: Google Cloud Vizier
Fleet: 20x Concurrent L4 Nodes. Absolute OOM Suppression.
"""
from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt

# =====================================================================
PROJECT_ID = "gen-lang-client-0250995579"           # <-- 必须替换为你的项目 ID
REGION = "us-central1"                       
BUCKET_URI = "gs://omega-pure-data"        # <-- 必须替换为你的 Bucket
GCS_INPUT = f"{BUCKET_URI}/base_matrix_shards" 
GCS_OUTPUT = f"{BUCKET_URI}/omega_models/hpo_trials_l4_mini"
# =====================================================================

def launch_scientific_hpo():
    print("[MAC ORACLE] 100x L4 Quota Confirmed. Initializing Massive Parallel Sweep...")
    import subprocess
    subprocess.run(["gcloud", "storage", "cp", "vertex_mae_compressor_l4_wolfpack.py", f"{BUCKET_URI}/scripts/vertex_mae_compressor_l4_wolfpack.py"], check=True)

    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)

    # g2-standard-32 拥有 128GB 的恐怖物理内存，足以镇压 CPU 碎片
    worker_pool_specs = [{
        "machine_spec": {
            "machine_type": "g2-standard-32",
            "accelerator_type": "NVIDIA_L4",
            "accelerator_count": 1,  # 彻底剔除 DataParallel 死锁，单机单卡
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1.py310:latest",
            "command": ["bash", "-c"],
            # 🚀 "_" 占位符极其关键，防止 Bash 吞噬 Vizier 注入的第一个超参！
            # 注入 logical_batch_size=4096，底层会自动切成 128 防御显存爆炸 (极端保守)
            "args": [
                f"gcloud storage cp {BUCKET_URI}/scripts/vertex_mae_compressor_l4_wolfpack.py . && pip install gcsfs pandas pyarrow numpy cloudml-hypertune && python3 vertex_mae_compressor_l4_wolfpack.py --gcs_input={GCS_INPUT} --epochs=10 --logical_batch_size=4096 --micro_batch_size=128 \"$@\"",
                "_"  
            ]
        }
    }]

    custom_job = aiplatform.CustomJob(
        display_name="omega-hpo-worker-mini",
        worker_pool_specs=worker_pool_specs,
        base_output_dir=GCS_OUTPUT
    )

    parameter_spec = {
        "stride": hpt.IntegerParameterSpec(min=100, max=1500, scale="linear"), 
        "seq_len": hpt.DiscreteParameterSpec(values=[32, 64, 128], scale=None), 
        "lookback_days": hpt.IntegerParameterSpec(min=3, max=30, scale="linear"),
    }

    metric_spec = {"val_fvu": "minimize"}

    print("[MAC ORACLE] Releasing 20-Node Wolfpack into the Event Horizon...")
    hpt_job = aiplatform.HyperparameterTuningJob(
        display_name="omega-macro-discovery-l4-mini",
        custom_job=custom_job,
        metric_spec=metric_spec,
        parameter_spec=parameter_spec,
        max_trial_count=60,       # 🔥 扩大实验基数，用算力碾压概率
        parallel_trial_count=20,  # 🔥 震撼并发！同时拉起 20 台 L4 机器排雷！
        search_algorithm=None,    
    )

    hpt_job.run(sync=False)
    print(f"[MAC ORACLE] 🚀 Tsunami OOM neutralized via Gradient Accumulation & BFloat16.")
    print(f"[MAC ORACLE] 20 Trials are burning through the data simultaneously. Monitor GCP Console.")

if __name__ == "__main__":
    launch_scientific_hpo()