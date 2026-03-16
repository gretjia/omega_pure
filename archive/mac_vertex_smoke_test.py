"""
THE OMEGA PROTOCOL: VERTEX IGNITION (SMOKE TEST)
Purpose: Cheap, fast dry-run to validate GCS IO, DataParallel, and Checkpointing before launching the 8x A100 beast.
"""

from google.cloud import aiplatform
import os

PROJECT_ID = "your-gcp-project-id"           # User will need to ensure auth is set
REGION = "us-central1"
BUCKET_URI = "gs://omega-pure-data"

GCS_INPUT = f"{BUCKET_URI}/base_matrix_shards"
GCS_OUTPUT = f"{BUCKET_URI}/omega_models/smoke_test"

def launch_smoke_test():
    print("[SMOKE TEST] Authenticating with Vertex AI...")
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)

    job = aiplatform.CustomTrainingJob(
        display_name="omega-mae-smoke-test",
        script_path="vertex_mae_compressor.py",
        container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1.py310:latest",
        requirements=["gcsfs", "pandas", "pyarrow", "numpy"],
    )

    print("[SMOKE TEST] Launching on a single cheap L4/T4 instance...")
    model = job.run(
        machine_type="g2-standard-4", # Extremely cheap L4 instance
        accelerator_type="NVIDIA_L4", 
        accelerator_count=1,
        args=[
            f"--gcs_input={GCS_INPUT}", 
            "--epochs=2",                # Only 2 epochs to test checkpoint recovery
            "--stride=400",
            "--batch_size=256",          # Small batch for single small GPU
            f"--checkpoint_dir={GCS_OUTPUT}/checkpoints",
            "--smoke_test_mode=true"     # Tells script to only read 1 file
        ],
        base_output_dir=GCS_OUTPUT,
        sync=False
    )
    print(f"[SMOKE TEST] Payload delivered. Check Vertex Console.")

if __name__ == "__main__":
    launch_smoke_test()
