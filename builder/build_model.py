import argparse
import torch
from diffusers import StableDiffusionXLPipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
    )
    pipe.save_pretrained(args.model_dir, revision="fp16")
