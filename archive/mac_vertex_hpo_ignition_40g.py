"""
THE OMEGA PROTOCOL: VERTEX IGNITION (THE SCIENTIFIC DISCOVERY)
Target: Google Cloud Vizier (Bayesian Optimization)
Strategy: 20x A100 40GB Parallel Sweet Spot (4 cards x 5 parallel trials)
"""
from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt

# =====================================================================
PROJECT_ID = "gen-lang-client-0250995579"           # <-- 必须替换
REGION = "us-central1"                       
BUCKET_URI = "gs://omega-pure-data"        # <-- 必须替换

GCS_INPUT = f"{BUCKET_URI}/base_matrix_shards" 
GCS_OUTPUT = f"{BUCKET_URI}/omega_models/hpo_trials_40g_sweetspot"
# =====================================================================

def launch_scientific_hpo():
    print("[MAC ORACLE] Uploading Compressor Payload to GCS...")
    import subprocess
    subprocess.run(["gcloud", "storage", "cp", "vertex_mae_compressor.py", f"{BUCKET_URI}/scripts/vertex_mae_compressor.py"], check=True)

    print("[MAC ORACLE] Authenticating with Vertex AI Vizier...")
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)

    # [战术 B：黄金分割] 4张A100(40G)为一个小队，同时派出5个小队，总计20卡满载
    worker_pool_specs = [{
        "machine_spec": {
            "machine_type": "a2-highgpu-4g",           # 4x A100 40GB
            "accelerator_type": "NVIDIA_TESLA_A100",
            "accelerator_count": 4,
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1.py310:latest",
            "command": ["bash", "-c"],
            # [防 OOM 核心装甲]: 强制压低 batch_size 到 1024 适配 40G 显存
            "args": [
                f"gcloud storage cp {BUCKET_URI}/scripts/vertex_mae_compressor.py . && pip install gcsfs pandas pyarrow numpy cloudml-hypertune && python3 vertex_mae_compressor.py --gcs_input={GCS_INPUT} --epochs=15 --batch_size=1024 \"$@\"",
                "_"  
            ]
        }
    }]

    custom_job = aiplatform.CustomJob(
        display_name="omega-macro-discovery-a100-40g",
        worker_pool_specs=worker_pool_specs,
        base_output_dir=GCS_OUTPUT
    )

    # 2. 猎杀范围的物理边界 (The Macro-Band Search Space)
    parameter_spec = {
        "stride": hpt.IntegerParameterSpec(min=100, max=1500, scale="linear"), 
        "seq_len": hpt.DiscreteParameterSpec(values=[32, 64, 128, 256], scale=None), 
        "lookback_days": hpt.IntegerParameterSpec(min=3, max=30, scale="linear"),
    }

    metric_spec = {"val_fvu": "minimize"}

    print("[MAC ORACLE] Initializing Bayesian Space (20-GPU Parallel Matrix)...")
    hpt_job = aiplatform.HyperparameterTuningJob(
        display_name="omega-macro-discovery-a100-40g-parallel",
        custom_job=custom_job,
        metric_spec=metric_spec,
        parameter_spec=parameter_spec,
        max_trial_count=30,       
        parallel_trial_count=5,   # [核心修改]: 5路并发，瞬间榨干20卡配额
        search_algorithm=None,    
    )

    hpt_job.run(sync=False)
    print(f"[MAC ORACLE] 🚀 Parallel Scientific Search initiated. (5 Concurrent Trials, 4x A100 each)")

if __name__ == "__main__":
    launch_scientific_hpo()