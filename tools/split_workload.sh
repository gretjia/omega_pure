#!/bin/bash
# Re-launch using the system python explicitly since we verified it has the torch+rocm installation in ~/.local/lib.

ssh linux1-lx "cd /home/zepher/omega_pure && export HSA_OVERRIDE_GFX_VERSION=11.0.0 && export HSA_ENABLE_SDMA=0 && nohup /usr/bin/python3 omega_tensor_materializer_pytorch_patched_v3.py --base_l1_dir /omega_pool/parquet_data/latest_base_l1/host=linux1/ --output_dir /home/zepher/omega_pure/base_matrix_shards_v2/host=linux1/ > materializer_pt_linux.log 2>&1 < /dev/null &"

ssh linux1-lx "cd /home/zepher/omega_pure && export HSA_OVERRIDE_GFX_VERSION=11.0.0 && export HSA_ENABLE_SDMA=0 && nohup /usr/bin/python3 omega_tensor_materializer_pytorch_patched_v3.py --base_l1_dir /omega_pool/parquet_data/latest_base_l1/host=windows1/ --output_dir /home/zepher/omega_pure/base_matrix_shards_v2/host=windows1/ > materializer_pt_windows.log 2>&1 < /dev/null &"

echo "V3 GPU Jobs Launched."
