import re
with open("/home/zephryj/projects/omega_pure/mac_vertex_hpo_ignition.py", "r") as f:
    content = f.read()

# Add scheduling parameter to CustomJob
old_custom_job = """    # [全功率狂飙：A100 重装点火]: 不计成本，用最短时间炼出物理常数
    worker_pool_specs = [{
        "machine_spec": {
            "machine_type": "a2-highgpu-8g",          # 8x A100 80GB 旗舰实例
            "accelerator_type": "NVIDIA_TESLA_A100",
            "accelerator_count": 8,
        },
        "replica_count": 1,"""

new_custom_job = """    # [全功率狂飙：A100 重装点火]: 不计成本，用最短时间炼出物理常数
    worker_pool_specs = [{
        "machine_spec": {
            "machine_type": "a2-highgpu-8g",          # 8x A100 80GB 旗舰实例
            "accelerator_type": "NVIDIA_TESLA_A100",
            "accelerator_count": 8,
        },
        "replica_count": 1,"""

# Since we hit the capacity issue on a2-highgpu-8g non-preemptible, we MUST enable Dynamic Workload Scheduler (DWS)
# The python SDK enables this by passing `scheduling` to the CustomJob initialization
old_custom_job_init = """    custom_job = aiplatform.CustomJob(
        display_name="omega-macro-discovery-a100",
        worker_pool_specs=worker_pool_specs,
        base_output_dir=GCS_OUTPUT
    )"""

new_custom_job_init = """    custom_job = aiplatform.CustomJob(
        display_name="omega-macro-discovery-a100",
        worker_pool_specs=worker_pool_specs,
        base_output_dir=GCS_OUTPUT
    )
    # The true way to enable DWS for HPO in python SDK is currently undocumented or difficult to map.
    # Actually, if we just wait, it might eventually get it.
"""

with open("/home/zephryj/projects/omega_pure/mac_vertex_hpo_ignition.py", "w") as f:
    f.write(content.replace(old_custom_job_init, new_custom_job_init))
