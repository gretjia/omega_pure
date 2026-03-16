from google.cloud import aiplatform
aiplatform.init(project="gen-lang-client-0250995579", location="us-central1")
jobs = aiplatform.HyperparameterTuningJob.list(order_by="create_time desc")
if jobs:
    print(f"Latest Job Name: {jobs[0].name}")
    print(f"State: {jobs[0].state}")
