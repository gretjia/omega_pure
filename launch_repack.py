from google.cloud import aiplatform
import time

PROJECT_ID = "gen-lang-client-0250995579"
REGION = "us-central1"
BUCKET_URI = "gs://omega-pure-data"

def launch_repack():
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)

    worker_pool_specs = [{
        "machine_spec": {
            "machine_type": "n1-highmem-8",
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1.py310:latest",
            "command": ["bash", "-c"],
            "args": [
                f"pip install polars gcsfs pyarrow tqdm && gcloud storage cp {BUCKET_URI}/scripts/cloud_repack_v5.py . && python3 cloud_repack_v5.py",
            ]
        }
    }]

    job = aiplatform.CustomJob(
        display_name="omega-data-repack-v5",
        worker_pool_specs=worker_pool_specs,
    )

    print("Launching Data Repackaging Job...")
    job.run(sync=False)
    
    # Wait a bit for resource to be created
    for _ in range(10):
        try:
            print(f"Job launched: {job.resource_name}")
            break
        except:
            time.sleep(2)

if __name__ == "__main__":
    import subprocess
    # Upload the repack script first
    subprocess.run(["gcloud", "storage", "cp", "cloud_repack_v5.py", f"{BUCKET_URI}/scripts/cloud_repack_v5.py"], check=True)
    launch_repack()
