import argparse
from diffusers import StableDiffusionXLPipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--model-dir", type=str, required=True)
    args = parser.parse_args()
    pipe = StableDiffusionXLPipeline.from_pretrained(args.model_name)
    pipe.save_pretrained(args.model_dir)
