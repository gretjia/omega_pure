#!/bin/bash
set -e

# Configuration
REGION="us-central1"
DISPLAY_NAME="omega-data-repack-v10-chronoforge-httpfs"
MACHINE_TYPE="n1-highmem-64" # 416GB RAM, 64 vCPUs
DISK_SIZE_GB=1000 # 1TB SSD for fast DuckDB spillage and Polars merge
IMAGE_URI="us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-14.py310:latest"

# 1. Ensure the latest script is on GCS
echo "[1/2] Uploading The Chrono-Forge to GCS..."
gcloud storage cp /home/zephryj/projects/omega_pure/tools/repack_to_ticker_shards.py gs://omega-pure-data/scripts/repack_to_ticker_shards.py

# 2. Create the dynamic YAML configuration
echo "[2/2] Generating Vertex AI Job Config..."
cat << YAML > /tmp/cloud_repack_job.yaml
workerPoolSpecs:
  - machineSpec:
      machineType: ${MACHINE_TYPE}
    diskSpec:
      bootDiskType: pd-ssd
      bootDiskSizeGb: ${DISK_SIZE_GB}
    replicaCount: 1
    containerSpec:
      imageUri: ${IMAGE_URI}
      command:
      - bash
      - -c
      args:
      - >
        echo "=== INSTALLING DEPENDENCIES ===" &&
        pip install --upgrade pip &&
        pip install duckdb polars pyarrow gcsfs fsspec &&
        echo "=== DOWNLOADING SCRIPT ===" &&
        gcloud storage cp gs://omega-pure-data/scripts/repack_to_ticker_shards.py . &&
        mkdir -p /tmp/duckdb_spill &&
        echo "=== IGNITING THE CHRONO-FORGE ===" &&
        python3 repack_to_ticker_shards.py --input_dir gs://omega-pure-data/base_matrix_shards --output_dir gs://omega-pure-data/ticker_matrix_shards_v5 --tmp_dir /tmp/duckdb_spill --memory_limit 350GB --threads 60
YAML

# 3. Submit the job
echo "Submitting Custom Job to Vertex AI..."
gcloud beta ai custom-jobs create \
    --region=${REGION} \
    --display-name=${DISPLAY_NAME} \
    --config=/tmp/cloud_repack_job.yaml

echo "Done! Monitor in Google Cloud Console."
