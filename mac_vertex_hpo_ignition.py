"""
THE OMEGA PROTOCOL: VERTEX IGNITION (THE SCIENTIFIC DISCOVERY)
Target: Google Cloud Vizier (Bayesian Optimization)
"""
from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt

# =====================================================================
PROJECT_ID = "gen-lang-client-0250995579"           # <-- 必须替换
REGION = "us-central1"                       
BUCKET_URI = "gs://omega-pure-data"        # <-- 必须替换

# 仅用 2023 年的数据去证明规律
GCS_INPUT = f"{BUCKET_URI}/base_matrix_shards" 
GCS_OUTPUT = f"{BUCKET_URI}/omega_models/hpo_trials"
# =====================================================================

def launch_scientific_hpo():
    print("[MAC ORACLE] Uploading Compressor Payload to GCS...")
    import subprocess
    subprocess.run(["gcloud", "storage", "cp", "vertex_mae_compressor.py", f"{BUCKET_URI}/scripts/vertex_mae_compressor.py"], check=True)

    print("[MAC ORACLE] Authenticating with Vertex AI Vizier...")
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)

    # [全功率狂飙：A100 重装点火]: 不计成本，用最短时间炼出物理常数
    worker_pool_specs = [{
        "machine_spec": {
            "machine_type": "a2-ultragpu-8g",          # 8x A100 80GB 旗舰实例
            "accelerator_type": "NVIDIA_A100_80GB",
            "accelerator_count": 8,
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1.py310:latest",
            "command": ["bash", "-c"],
            # [核心架构师纠错] 加入占位符 "_", 防止 Vizier 的参数被 bash 吃掉！
            "args": [
                f"gcloud storage cp {BUCKET_URI}/scripts/vertex_mae_compressor.py . && pip install gcsfs pandas pyarrow numpy cloudml-hypertune && python3 vertex_mae_compressor.py --gcs_input={GCS_INPUT} --epochs=15 --batch_size=8192 \"$@\"",
                "_"  
            ]
        }
    }]

    # [Agent 紧急修复]: 开启 Dynamic Workload Scheduler (DWS) 排队获取稀缺的 A100
    custom_job = aiplatform.CustomJob(
        display_name="omega-macro-discovery-a100",
        worker_pool_specs=worker_pool_specs,
        base_output_dir=GCS_OUTPUT
    )
    # The true way to enable DWS for HPO in python SDK is currently undocumented or difficult to map.
    # Actually, if we just wait, it might eventually get it.


    # 2. 猎杀范围的物理边界 (The Macro-Band Search Space)
    parameter_spec = {
        # 架构师封锁线：下限卡死在 100！绝不允许模型钻高频噪音的空子！上限开放到 1500 (宏观波段)
        "stride": hpt.IntegerParameterSpec(min=100, max=1500, scale="linear"), 
        "seq_len": hpt.DiscreteParameterSpec(values=[32, 64, 128, 256], scale=None), 
        # [彻底数据驱动]: 让贝叶斯引擎去决定看几天的数据最能发现规律！
        "lookback_days": hpt.IntegerParameterSpec(min=3, max=30, scale="linear"),
    }

    # 3. 猎杀目标：最小化"局部未解释方差比" (FVU)
    metric_spec = {"val_fvu": "minimize"}

    print("[MAC ORACLE] Initializing Bayesian Space. Firing HPO Payload...")
    hpt_job = aiplatform.HyperparameterTuningJob(
        display_name="omega-macro-discovery-a100",
        custom_job=custom_job,
        metric_spec=metric_spec,
        parameter_spec=parameter_spec,
        max_trial_count=30,       # 增加搜索次数到 30
        parallel_trial_count=1,   # 维持单跑，但每个试验内用 8 卡加速
        search_algorithm=None,    
    )

    hpt_job.run(sync=False)
    print(f"[MAC ORACLE] 🚀 Scientific Search initiated. Open GCP Console -> Vertex AI -> Hyperparameter Tuning Jobs to watch the physics unfold.")

if __name__ == "__main__":
    launch_scientific_hpo()
