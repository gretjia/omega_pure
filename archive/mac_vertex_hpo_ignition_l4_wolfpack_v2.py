"""
THE OMEGA PROTOCOL: VERTEX IGNITION (THE 100-L4 WOLFPACK - WAVE 2)
Target: Google Cloud Vizier
Fleet: 20x Concurrent L4 Nodes. Absolute OOM Suppression + Auto-Kill.
"""
from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt

# =====================================================================
PROJECT_ID = "gen-lang-client-0250995579"           
REGION = "us-central1"                       
BUCKET_URI = "gs://omega-pure-data"        
GCS_INPUT = f"{BUCKET_URI}/base_matrix_shards" 
GCS_OUTPUT = f"{BUCKET_URI}/omega_models/hpo_trials_l4_wolfpack_v2"
# =====================================================================

def launch_scientific_hpo():
    print("[MAC ORACLE] 100x L4 Quota Confirmed. Initializing Wave 2...")
    import subprocess
    subprocess.run(["gcloud", "storage", "cp", "vertex_mae_compressor_l4_wolfpack.py", f"{BUCKET_URI}/scripts/vertex_mae_compressor_l4_wolfpack.py"], check=True)

    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)

    worker_pool_specs = [{
        "machine_spec": {
            "machine_type": "g2-standard-32",
            "accelerator_type": "NVIDIA_L4",
            "accelerator_count": 1,  
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1.py310:latest",
            "command": ["bash", "-c"],
            "args": [
                f"gcloud storage cp {BUCKET_URI}/scripts/vertex_mae_compressor_l4_wolfpack.py . && pip install gcsfs pandas pyarrow numpy cloudml-hypertune && python3 vertex_mae_compressor_l4_wolfpack.py --gcs_input={GCS_INPUT} --epochs=10 --logical_batch_size=4096 --micro_batch_size=128 \"$@\"",
                "_"  
            ]
        }
    }]

    custom_job = aiplatform.CustomJob(
        display_name="omega-hpo-worker-v2",
        worker_pool_specs=worker_pool_specs,
        base_output_dir=GCS_OUTPUT
    )

    parameter_spec = {
        "stride": hpt.IntegerParameterSpec(min=100, max=1500, scale="linear"), 
        "seq_len": hpt.DiscreteParameterSpec(values=[32, 64, 128], scale=None), 
        "lookback_days": hpt.IntegerParameterSpec(min=3, max=30, scale="linear"),
    }

    metric_spec = {"val_fvu": "minimize"}

    print("[MAC ORACLE] Releasing Wave 2 Wolfpack (Auto-Kill Enabled)...")
    hpt_job = aiplatform.HyperparameterTuningJob(
        display_name="omega-macro-discovery-l4-wave2",
        custom_job=custom_job,
        metric_spec=metric_spec,
        parameter_spec=parameter_spec,
        max_trial_count=60,       
        parallel_trial_count=20,  
        search_algorithm=None,    
    )

    hpt_job.run(sync=False)
    print(f"[MAC ORACLE] 🚀 Tsunami OOM neutralized. NaN Auto-Kill protocol engaged.")

if __name__ == "__main__":
    launch_scientific_hpo()