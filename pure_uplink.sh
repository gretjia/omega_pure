#!/bin/bash
# THE OMEGA PROTOCOL: PURE MEMORY UPLINK (Fix)
# Streams using rsync/scp to local buffer sequentially and uploads with explicit timeout to avoid hanging pipes.

GCS_BUCKET="gs://omega-pure-data/base_matrix_shards"
TARGET="linux1-lx"
REMOTE_DIR="/home/zepher/omega_pure/base_matrix_shards_2024"
JUMP="hk-node"
TMP_DIR="/tmp/omega_chunk"

echo "============================================================"
echo "🛸 OMEGA PROTOCOL: THE LOCAL-BUFFER PIPE"
echo "============================================================"

# Get the list of parquet files
echo "📡 Scanning target node for shards..."
FILES=$(ssh -J $JUMP -o ConnectTimeout=10 $TARGET "ls -1 $REMOTE_DIR/*.parquet" || echo "FAILED")

if [[ "$FILES" == *"FAILED"* ]]; then
    echo "❌ [FATAL] Cannot connect to $TARGET via SSH."
    exit 1
fi

TOTAL=$(echo "$FILES" | wc -l)
echo "🎯 Found $TOTAL shards. Starting secure transport..."

mkdir -p $TMP_DIR
COUNT=0
for FILE in $FILES; do
    COUNT=$((COUNT+1))
    BASENAME=$(basename "$FILE")
    
    if gcloud storage ls "$GCS_BUCKET/$BASENAME" >/dev/null 2>&1; then
        echo "⏩ [SKIP] $BASENAME already exists in GCP ($COUNT/$TOTAL)"
        continue
    fi

    echo "⏳ [PULLING] $BASENAME ($COUNT/$TOTAL) ..."
    
    # Safely pull one file locally first. It's only 200MB, perfectly fine for the 50G disk.
    scp -o ProxyJump=$JUMP -o ConnectTimeout=15 "$TARGET:$FILE" "$TMP_DIR/$BASENAME"
    
    if [ -f "$TMP_DIR/$BASENAME" ]; then
        echo "   -> [UPLOADING] to GCP..."
        gcloud storage cp "$TMP_DIR/$BASENAME" "$GCS_BUCKET/" > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo "✔️  [SUCCESS] $BASENAME completed."
            rm -f "$TMP_DIR/$BASENAME"
        else
            echo "❌  [FAILED GCP UPLOAD] for $BASENAME."
        fi
    else
        echo "❌  [FAILED SCP PULL] for $BASENAME."
    fi
done
echo "============================================================"
echo "🏁 UPLINK COMPLETE."
echo "============================================================"