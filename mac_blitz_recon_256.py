"""
THE OMEGA PROTOCOL: BLITZKRIEG V3 IGNITION (RECONNAISSANCE - 256 SEQ)
Target: Test boundary of long-range dependencies (Seq Len 256).
Fleet: 10x Concurrent STANDARD L4 Nodes (g2-standard-8).
Action: Narrow Stride search around current winners with doubled Seq Len.
"""
from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt
import subprocess

PROJECT_ID = "gen-lang-client-0250995579"      
REGION = "us-central1"
BUCKET_URI = "gs://omega-pure-data"        
GCS_INPUT = f"{BUCKET_URI}/base_matrix_shards"
GCS_OUTPUT = f"{BUCKET_URI}/omega_models/hpo_blitz_recon_256"

def launch_recon_256():
    print("[MAC ORACLE] Initializing 256-Seq Reconnaissance Mission...")
    
    # We use the existing V3 core script
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)
    
    worker_pool_specs = [{
        "machine_spec": {
            "machine_type": "g2-standard-8", # Use 32GB RAM to handle larger Seq Len safely
            "accelerator_type": "NVIDIA_L4",
            "accelerator_count": 1, 
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1.py310:latest",
            "command": ["bash", "-c"],
            "args": [
                f"gcloud storage cp {BUCKET_URI}/scripts/vertex_mae_blitz_v3.py . && pip install gcsfs pandas pyarrow numpy cloudml-hypertune && python3 vertex_mae_blitz_v3.py --gcs_input={GCS_INPUT} --logical_batch_size=4096 --micro_batch_size=256 --max_steps=3000 --report_freq=500 \"$@\"",
                "_"  
            ]
        },
        "disk_spec": {
            "boot_disk_type": "pd-ssd",
            "boot_disk_size_gb": 100
        }
    }]
    
    custom_job = aiplatform.CustomJob(
        display_name="omega-blitz-v3-recon-256-worker",
        worker_pool_specs=worker_pool_specs,
        base_output_dir=GCS_OUTPUT
    )
    
    # 🎯 Focused search around successful Stride range from the main job
    parameter_spec = {
        "stride": hpt.IntegerParameterSpec(min=700, max=1400, scale="linear"), 
        "seq_len": hpt.DiscreteParameterSpec(values=[256], scale=None), 
    }
    
    print("[MAC ORACLE] Unleashing 10-Node RECON Wolfpack...")
    hpt_job = aiplatform.HyperparameterTuningJob(
        display_name="omega-hpo-recon-256-standard",
        custom_job=custom_job,
        metric_spec={"val_fvu": "minimize"},
        parameter_spec=parameter_spec,
        search_algorithm=None,
        max_trial_count=20,      
        parallel_trial_count=10, 
    )
    
    hpt_job.run(sync=False)
    print(f"[MAC ORACLE] 🚀 Recon 256 Mission Deployed. Resource Name: {hpt_job.resource_name}")

if __name__ == "__main__":
    launch_recon_256()
