"""
THE OMEGA PROTOCOL: BLITZKRIEG V3 FINAL RECON (256/512 SEQ)
Target: Pinpoint the best Stride for long horizons.
Fleet: 50x Concurrent STANDARD L4 Nodes (g2-standard-8).
Action: Narrow search [900-1500] with Seq [256, 512]. Fixed Suicide Logic.
"""
from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt
import subprocess

PROJECT_ID = "gen-lang-client-0250995579"      
REGION = "us-central1"
BUCKET_URI = "gs://omega-pure-data"        
GCS_INPUT = f"{BUCKET_URI}/base_matrix_shards"
GCS_OUTPUT = f"{BUCKET_URI}/omega_models/hpo_blitz_final_recon"

def launch_final_recon():
    print("[MAC ORACLE] Initializing Final Phase 2 Reconnaissance (256/512 Seq)...")
    
    # Sync fixed core script
    subprocess.run(["gcloud", "storage", "cp", "vertex_mae_blitz_v3.py", 
                    f"{BUCKET_URI}/scripts/vertex_mae_blitz_v3.py"], check=True)
    
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)
    
    # Use standard-8 for larger memory headroom with 512 Seq
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
                f"gcloud storage cp {BUCKET_URI}/scripts/vertex_mae_blitz_v3.py . && pip install gcsfs pandas pyarrow numpy cloudml-hypertune && python3 vertex_mae_blitz_v3.py --gcs_input={GCS_INPUT} --logical_batch_size=4096 --micro_batch_size=128 --max_steps=3000 --report_freq=500 \"$@\"",
                "_"  
            ]
        },
        "disk_spec": {
            "boot_disk_type": "pd-ssd",
            "boot_disk_size_gb": 100
        }
    }]
    
    custom_job = aiplatform.CustomJob(
        display_name="omega-blitz-v3-final-recon-worker",
        worker_pool_specs=worker_pool_specs,
        base_output_dir=GCS_OUTPUT
    )
    
    # 🎯 Targeting the discovered sweet spot
    parameter_spec = {
        "stride": hpt.IntegerParameterSpec(min=900, max=1500, scale="linear"), 
        "seq_len": hpt.DiscreteParameterSpec(values=[256, 512], scale=None), 
    }
    
    print("[MAC ORACLE] Unleashing 50-Node FINAL RECON Wolfpack...")
    hpt_job = aiplatform.HyperparameterTuningJob(
        display_name="omega-hpo-final-recon-2025-standard",
        custom_job=custom_job,
        metric_spec={"val_fvu": "minimize"},
        parameter_spec=parameter_spec,
        search_algorithm=None,
        max_trial_count=100,      
        parallel_trial_count=50, 
    )
    
    hpt_job.run(sync=False)
    # Wait for ID to appear
    import time
    time.sleep(10)
    print(f"[MAC ORACLE] 🚀 Final Recon Deployed. Resource Name: {hpt_job.resource_name}")

if __name__ == "__main__":
    launch_final_recon()
