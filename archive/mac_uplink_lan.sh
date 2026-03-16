#!/bin/bash
# THE OMEGA PROTOCOL: THE LOCAL WORMHOLE UPLINK (LAN + Rsync)
# Executes ON THE LINUX NODE, routing GCP upload traffic directly through the Mac's LAN Proxy (192.168.3.93:7897).

GCS_BUCKET="gs://omega-pure-data/base_matrix_shards"
LOCAL_DIR="/home/zepher/omega_pure/base_matrix_shards_2024"
PROXY="http://192.168.3.93:7897"

echo "============================================================"
echo "🛸 OMEGA PROTOCOL: THE LOCAL WORMHOLE UPLINK"
echo "============================================================"
echo "Routing traffic through Mac LAN proxy: $PROXY"

export HTTPS_PROXY=$PROXY
export HTTP_PROXY=$PROXY
export ALL_PROXY=$PROXY

# Using gsutil rsync instead of cp. Rsync natively checks what is already in the bucket
# and only uploads missing or changed files. It handles resuming automatically.
echo "⏳ [BEAMING] Initiating hyper-speed parallel rsync from Linux directly to GCP..."
/home/zepher/google-cloud-sdk/bin/gcloud storage rsync -R $LOCAL_DIR $GCS_BUCKET

if [ $? -eq 0 ]; then
    echo "✔️ [SUCCESS] The Wormhole transfer is complete."
else
    echo "❌ [ERROR] The Wormhole encountered an anomaly."
fi