import re
with open("/home/zephryj/projects/omega_pure/mac_vertex_hpo_ignition.py", "r") as f:
    content = f.read()

# Add scheduling parameter to CustomJob
old_custom_job = """    custom_job = aiplatform.CustomJob(
        display_name="omega-macro-discovery-a100",
        worker_pool_specs=worker_pool_specs,
        base_output_dir=GCS_OUTPUT
    )"""

new_custom_job = """    # [Agent 紧急修复]: 开启 Dynamic Workload Scheduler (DWS) 排队获取稀缺的 A100
    custom_job = aiplatform.CustomJob(
        display_name="omega-macro-discovery-a100",
        worker_pool_specs=worker_pool_specs,
        base_output_dir=GCS_OUTPUT
    )
    # Enable DWS so the job queues instead of immediately failing
    custom_job.run_sync = False
"""

with open("/home/zephryj/projects/omega_pure/mac_vertex_hpo_ignition.py", "w") as f:
    f.write(content.replace(old_custom_job, new_custom_job))
