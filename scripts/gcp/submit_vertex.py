"""
Vertex AI Job Submission Script for SciMLx.
Launches a Custom Container training job on a GPU-enabled worker.
"""

import argparse
from google.cloud import aiplatform

def submit_job(
    project_id: str,
    region: str,
    image_uri: str,
    display_name: str,
    machine_type: str = "n1-standard-8",
    accelerator_type: str = "NVIDIA_TESLA_T4",
    accelerator_count: int = 1,
    args: list = None
):
    aiplatform.init(project=project_id, location=region)

    job = aiplatform.CustomContainerTrainingJob(
        display_name=display_name,
        container_uri=image_uri,
    )

    model = job.run(
        args=args or [],
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        replica_count=1,
    )

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="GCP Project ID")
    parser.add_argument("--region", default="us-central1", help="GCP Region")
    parser.add_argument("--image", required=True, help="Artifact Registry image URI")
    parser.add_argument("--name", default="scimlx-training-cuda", help="Job display name")
    parser.add_argument("--gpu-type", default="NVIDIA_TESLA_T4", help="e.g. NVIDIA_L4, NVIDIA_TESLA_A100_40GB")
    parser.add_argument("--gpu-count", type=int, default=1)
    
    # Capture all remaining args to pass to train.py
    parsed, unknown = parser.parse_known_args()
    
    print(f"Submitting job '{parsed.name}' to Vertex AI...")
    submit_job(
        project_id=parsed.project,
        region=parsed.region,
        image_uri=parsed.image,
        display_name=parsed.name,
        accelerator_type=parsed.gpu_type,
        accelerator_count=parsed.gpu_count,
        args=unknown
    )
