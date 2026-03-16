#!/bin/bash
set -e

# Sync from Windows (100.123.90.25) D:\Omega_frames\latest_feature_l2\host=windows1 
# to Linux (100.64.97.113) /omega_pool/parquet_data/latest_feature_l2/host=windows1

# The execution strategy as defined in GEMINI.md for large MTU transfers:
# 1. Pull from Windows to Mac-Back (or Omega-VM if routed correctly)
# 2. Push to Linux
# Or since we are on Omega-VM with hk-wg active, we can try direct rsync over the hk-wg ssh paths first, 
# but the most reliable way avoiding MTU stalls mentioned in the prompt is via Mac-Back. 
# However, I have direct SSH aliases `linux1-lx` and `windows1-w1` working perfectly right now.

echo "Creating target directory on linux1-lx..."
ssh linux1-lx "mkdir -p /omega_pool/parquet_data/latest_feature_l2/host=windows1"

echo "Initiating direct pull from windows1-w1 to omega-vm, then push to linux1-lx..."
# We will use scp over the existing working SSH connections.
mkdir -p /tmp/win_l2_sync
scp -r windows1-w1:'D:/Omega_frames/latest_feature_l2/host=windows1/*.parquet' /tmp/win_l2_sync/
scp /tmp/win_l2_sync/*.parquet linux1-lx:/omega_pool/parquet_data/latest_feature_l2/host=windows1/
rm -rf /tmp/win_l2_sync

echo "Sync complete."
