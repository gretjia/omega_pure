#!/bin/bash
# THE OMEGA PROTOCOL: THE MAC WORMHOLE UPLINK
# Executes ON THE LINUX NODE, routing GCP upload traffic directly through the Mac's Tailscale Proxy (100.72.87.94:7897).
# Bypasses the omega-vm bottleneck entirely.

GCS_BUCKET="gs://omega-pure-data/base_matrix_shards"
LOCAL_DIR="/home/zepher/omega_pure/base_matrix_shards_2024"
PROXY="http://100.72.87.94:7897"

echo "============================================================"
echo "🛸 OMEGA PROTOCOL: THE WORMHOLE (MAC PROXY UPLINK)"
echo "============================================================"
echo "Routing traffic through Mac Studio proxy: $PROXY"

# Set proxies for gcloud
export HTTPS_PROXY=$PROXY
export HTTP_PROXY=$PROXY
export ALL_PROXY=$PROXY

# Use gsutil rsync for the fastest, most reliable directory synchronization
echo "⏳ [BEAMING] Initiating hyper-speed rsync from Linux to GCP..."
gcloud storage rsync -R $LOCAL_DIR $GCS_BUCKET

if [ $? -eq 0 ]; then
    echo "✔️ [SUCCESS] The Wormhole transfer is complete."
else
    echo "❌ [ERROR] The Wormhole encountered an anomaly."
fi