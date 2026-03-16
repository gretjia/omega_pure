import os
import argparse
from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt

PROJECT_ID = "gen-lang-client-0250995579"           
REGION = "us-central1"                       
BUCKET_URI = "gs://omega-pure-data"        

GCS_INPUT = f"{BUCKET_URI}/base_matrix_shards" 
GCS_OUTPUT = f"{BUCKET_URI}/omega_models/hpo_trials"

def launch_scientific_hpo():
    print("[MAC ORACLE] Authenticating with Vertex AI Vizier...")
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)

    worker_pool_specs = [{
        "machine_spec": {
            "machine_type": "a2-ultragpu-8g",          
            "accelerator_type": "NVIDIA_TESLA_A100",
            "accelerator_count": 8,
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1.py310:latest",
            "command": ["bash", "-c"],
            "args": [
                f"echo 'test' \"$@\"",
                "_"  
            ]
        }
    }]

    custom_job = aiplatform.CustomJob(
        display_name="omega-macro-discovery-a100-test",
        worker_pool_specs=worker_pool_specs,
        base_output_dir=GCS_OUTPUT
    )

    parameter_spec = {
        "stride": hpt.IntegerParameterSpec(min=100, max=1500, scale="linear"), 
        "seq_len": hpt.DiscreteParameterSpec(values=[32, 64], scale=None), 
    }

    metric_spec = {"val_fvu": "minimize"}

    hpt_job = aiplatform.HyperparameterTuningJob(
        display_name="omega-macro-discovery-a100-test",
        custom_job=custom_job,
        metric_spec=metric_spec,
        parameter_spec=parameter_spec,
        max_trial_count=2,       
        parallel_trial_count=1,   
        search_algorithm=None,    
    )

    try:
        # Run sync to catch any immediate API exceptions
        hpt_job.run(sync=True)
    except Exception as e:
        print(f"Exception caught during run: {e}")
        
if __name__ == "__main__":
    launch_scientific_hpo()
