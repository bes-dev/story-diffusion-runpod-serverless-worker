import io
import os
import argparse
# runpod utils
import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils.rp_upload import upload_in_memory_object
from runpod.serverless.utils import rp_download, rp_cleanup
# predictor
import torch
from comic_generator_xl import ComicGeneratorXL
from rp_schema import INPUT_SCHEMA
# utils
from utils import compress_images_to_zip


# Worker params
model_dir = os.getenv("WORKER_MODEL_DIR", "/model")
id_length = int(os.getenv("WORKER_ID_LENGTH", 4))
total_length = int(os.getenv("WORKER_TOTAL_LENGTH", 5))
device = "cuda" if os.getenv("WORKER_USE_CUDA") == "True" else "cpu"
scheduler_type = os.getenv("WORKER_SCHEDULER_TYPE", "euler").lower()


def upload_result(result: io.BytesIO, key: str) -> str:
    """ Uploads result to S3 bucket if it is available, otherwise returns base64 encoded file. """
    # Upload to S3
    if os.environ.get('BUCKET_ENDPOINT_URL', False):
        return upload_in_memory_object(
            key,
            result.read(),
            bucket_creds = {
                "endpointUrl": os.environ.get('BUCKET_ENDPOINT_URL', None),
                "accessId": os.environ.get('BUCKET_ACCESS_KEY_ID', None),
                "accessSecret": os.environ.get('BUCKET_SECRET_ACCESS_KEY', None)
            }
        )
    # Base64 encode
    return result.decode('UTF-8')


def run(job):
    job_input = job['input']

    # Input validation
    validated_input = validate(job_input, INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    validated_input = validated_input['validated_input']

    # Inference image generator
    images = MODEL(
        prompts=validated_input["prompts"],
        negative_prompts=validated_input.get("negative_prompts", None),
        width=validated_input.get("width", 768),
        height=validated_input.get("height", 768),
        sa32=validated_input.get("sa32", 0.5),
        sa64=validated_input.get("sa64", 0.5),
        guidance_scale=validated_input.get("guidance_scale", 5.0),
        num_inference_steps=validated_input.get("num_inference_steps", 25),
        seed=validated_input.get("seed", 42)
    )

    # Upload output object
    zip_data = compress_images_to_zip(images)
    output_data = upload_audio(zip_data, f"{job['id']}.zip")
    job_output = {
        "output_data": output_data
    }

    # Remove downloaded input objects
    rp_cleanup.clean(['input_objects'])

    return job_output


if __name__ == "__main__":
    MODEL = ComicGeneratorXL(
        model_name=model_dir,
        id_length=id_length,
        total_length=total_length,
        device=device,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        scheduler_type=scheduler_type
    )

    runpod.serverless.start({"handler": run})
