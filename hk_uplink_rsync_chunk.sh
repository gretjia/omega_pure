#!/bin/bash
# Executes on the HK node directly
# Uses rsync for extremely resilient transfer from Linux1 -> HK local buffer (limited size), then uploads to GCP and cleans up.

LOCAL_DIR="/tmp/omega_sync_buffer"
GCS_BUCKET="gs://omega-pure-data/base_matrix_shards"
TARGET="zepher@100.64.97.113"
REMOTE_DIR="/home/zepher/omega_pure/base_matrix_shards_2024"
KEY="~/.ssh/id_ed25519_omega_workers"

mkdir -p $LOCAL_DIR

echo "============================================================"
echo "🛸 OMEGA PROTOCOL: HK DIRECT RSYNC UPLINK (BUFFERED)"
echo "============================================================"

echo "📡 Scanning target node for shards..."
FILES=$(ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=10 $TARGET "ls -1 $REMOTE_DIR/*.parquet")
TOTAL=$(echo "$FILES" | wc -l)
echo "🎯 Found $TOTAL shards. Starting buffered transport..."

COUNT=0
for FILE in $FILES; do
    COUNT=$((COUNT+1))
    BASENAME=$(basename "$FILE")
    
    # Check if already in GCS to allow resuming
    if /home/ubuntu/google-cloud-sdk/bin/gcloud storage ls "$GCS_BUCKET/$BASENAME" >/dev/null 2>&1; then
        echo "⏩ [SKIP] $BASENAME already exists in GCP ($COUNT/$TOTAL)"
        continue
    fi

    echo "⏳ [PULLING] $BASENAME ($COUNT/$TOTAL) ..."
    
    # Download single file using rsync to handle disconnects perfectly
    rsync -q --partial -e "ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=15" $TARGET:$FILE $LOCAL_DIR/$BASENAME
    
    if [ -f "$LOCAL_DIR/$BASENAME" ]; then
        echo "   -> [UPLOADING] to GCP..."
        /home/ubuntu/google-cloud-sdk/bin/gcloud storage cp "$LOCAL_DIR/$BASENAME" "$GCS_BUCKET/" >/dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo "✔️  [SUCCESS] $BASENAME completed."
            rm -f "$LOCAL_DIR/$BASENAME"
        else
            echo "❌  [FAILED GCP UPLOAD] for $BASENAME."
        fi
    else
        echo "❌  [FAILED RSYNC PULL] for $BASENAME."
    fi
done

echo "============================================================"
echo "🏁 HK DIRECT UPLINK COMPLETE."
echo "============================================================"