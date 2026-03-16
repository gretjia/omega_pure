"""
THE OMEGA PROTOCOL: VERTEX IGNITION
Execution Target: Omega VM (Central Control)
Action: Deploys the Epiplexity Compressor payload to GCP using 8x A100 Preemptible instances.
"""

from google.cloud import aiplatform

# =====================================================================
# TALEBIAN CONFIGURATION (Replace with your actual GCS parameters)
# =====================================================================
PROJECT_ID = "your-gcp-project-id"           # <-- 填入你的 GCP Project ID
REGION = "us-central1"                       # <-- 你的计算可用区
BUCKET_URI = "gs://omega-pure-data"          # <-- 填入你的主 Bucket 路径

GCS_INPUT = f"{BUCKET_URI}/base_matrix_shards"
GCS_OUTPUT = f"{BUCKET_URI}/omega_models/mae_oracle"

def launch_vertex_forge():
    print("[OMEGA VM] Authenticating with Google Cloud Vertex AI...")
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)

    print("[OMEGA VM] Forging Custom PyTorch Payload for 8x A100...")
    # Using Google's pure, pre-built PyTorch container. Zero Dockerfile entropy.
    job = aiplatform.CustomTrainingJob(
        display_name="omega-epiplexity-mae-8xa100-spot",
        script_path="vertex_mae_compressor.py", 
        container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1.py310:latest",
        requirements=["gcsfs", "pandas", "pyarrow", "numpy"],
    )

    print("[OMEGA VM] Firing Payload to NVIDIA A100 Cloud Cluster (8x A100 40GB, SPOT)...")
    # a2-highgpu-8g utilizes 8 NVIDIA A100 GPUs (40GB each)
    model = job.run(
        machine_type="a2-highgpu-8g",
        accelerator_type="NVIDIA_TESLA_A100", 
        accelerator_count=8,
        args=[
            f"--gcs_input={GCS_INPUT}", 
            "--epochs=50", 
            "--stride=400",
            "--batch_size=8192", # Greatly increased for 8x A100 (40GB x 8)
            f"--checkpoint_dir={GCS_OUTPUT}/checkpoints"
        ],
        base_output_dir=GCS_OUTPUT,
        scheduling_strategy="SPOT", # Enables preemptible Spot instance (60-70% cost reduction)
        sync=False # Asynchronous execution so omega-vm isn't held hostage
    )
    
    print(f"[OMEGA VM] Payload delivered. Vertex is establishing the Baseline Entropy.")

if __name__ == "__main__":
    launch_vertex_forge()
