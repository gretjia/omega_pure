#!/bin/bash
# Executes on the HK node directly
# Uses rsync for extremely resilient transfer from Linux1 -> HK, and then gcloud from HK -> GCP

LOCAL_DIR="/tmp/omega_sync_buffer"
GCS_BUCKET="gs://omega-pure-data/base_matrix_shards"
TARGET="zepher@100.64.97.113"
REMOTE_DIR="/home/zepher/omega_pure/base_matrix_shards_2024"
KEY="~/.ssh/id_ed25519_omega_workers"

mkdir -p $LOCAL_DIR

echo "============================================================"
echo "🛸 OMEGA PROTOCOL: HK DIRECT RSYNC UPLINK"
echo "============================================================"

# Since HK node disk is only 30GB free and total payload is 188GB, we MUST loop and sync a few at a time, OR 
# we can just use the memory pipe directly on HK node which is perfectly stable since it doesn't cross the GFW twice.

echo "📡 Scanning target node for shards..."
FILES=$(ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=10 $TARGET "ls -1 $REMOTE_DIR/*.parquet")
TOTAL=$(echo "$FILES" | wc -l)
echo "🎯 Found $TOTAL shards. Establishing direct memory pipe from HK..."

COUNT=0
for FILE in $FILES; do
    COUNT=$((COUNT+1))
    BASENAME=$(basename "$FILE")
    
    # Check if already in GCS to allow resuming
    if /home/ubuntu/google-cloud-sdk/bin/gcloud storage ls "$GCS_BUCKET/$BASENAME" >/dev/null 2>&1; then
        echo "⏩ [SKIP] $BASENAME already exists in GCP ($COUNT/$TOTAL)"
        continue
    fi

    echo "⏳ [BEAMING] $BASENAME ($COUNT/$TOTAL) ..."
    
    # The ultimate solution: HK node runs the SSH pipe directly to GCP! 
    # Because HK is outside the GFW, the connection to GCP is 200M pure fiber.
    # Because HK connects directly to Shenzhen via Tailscale (UDP hole punching), it's highly resilient.
    ssh -i $KEY -o StrictHostKeyChecking=no -o ServerAliveInterval=15 $TARGET "cat $FILE" | /home/ubuntu/google-cloud-sdk/bin/gcloud storage cp - "$GCS_BUCKET/$BASENAME"
    
    if [ ${PIPESTATUS[1]} -eq 0 ]; then
        echo "✔️  [SUCCESS] $BASENAME uploaded."
    else
        echo "❌  [FAILED] Upload broken for $BASENAME. Retrying next round."
    fi
done

echo "============================================================"
echo "🏁 HK DIRECT UPLINK COMPLETE."
echo "============================================================"