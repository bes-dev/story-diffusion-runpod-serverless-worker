# StoryDiffusion: Serverless RunPod Worker

## RunPod Endpoint

This repository contains the worker for the StoryDiffusion AI Endpoints.

## Docker Image

```bash
docker build .
```
 or

 ```bash
 docker pull devbes/story-diffusion-serverless-worker:latest
 ```

## Environment Variables

### S3 storage

- BUCKET_ENDPOINT_URL
- BUCKET_ACCESS_KEY_ID
- BUCKET_SECRET_ACCESS_KEY

### Dockerfile configuration

- WORKER_MODEL_NAME (default = SG161222/RealVisXL_V4.0)
- WORKER_ID_LENGTH (default = 4)
- WORKER_TOTAL_LENGTH (default = 5)
- WORKER_SCHEDULER_TYPE (default = euler)


## Continuous Deployment
This worker follows a modified version of the [worker template](https://github.com/runpod-workers/worker-template) where the Docker build workflow contains additional SD models to be built and pushed.

## API

Use 'img' as a trigger word for personalized generation cases.

```json
{
  "input": {
      "prompts": [<prompt1:str>, <promptn:str>],
      "negative_prompt": <negative_prompt: str>,
      "width": <width:int>,
      "height": <height:int>,
      "sa32": <sa32:float>,
      "sa64": <sa64:float>,
      "guidance_scale": <guidance_scale:float>,
      "num_inference_steps": <num_inference_steps:int>,
      "seed": <seed:int>,
      "image_ref": <link to reference image (Optional):str>
  }
}
```

Sample request:
```json
{
  "input": {
      "prompts": ["Harold img is a curious and clever boy with bright blue eyes and messy brown hair. He always wears a red hat and carries a tiny backpack full of gadgets. discovering a golden key in his grandmother's attic.", "Harold img is a curious and clever boy with bright blue eyes and messy brown hair. He always wears a red hat and carries a tiny backpack full of gadgets. talking to a squirrel in a magical forest.", "Harold img is a curious and clever boy with bright blue eyes and messy brown hair. He always wears a red hat and carries a tiny backpack full of gadgets. jumping on giant marshmallows at the top of a mountain.", "Harold img is a curious and clever boy with bright blue eyes and messy brown hair. He always wears a red hat and carries a tiny backpack full of gadgets. hopping into a boat on a sparkling river.", "Harold img is a curious and clever boy with bright blue eyes and messy brown hair. He always wears a red hat and carries a tiny backpack full of gadgets. talking to a mole in an underground cave.", "Harold img is a curious and clever boy with bright blue eyes and messy brown hair. He always wears a red hat and carries a tiny backpack full of gadgets. dancing at a village festival.", "Harold img is a curious and clever boy with bright blue eyes and messy brown hair. He always wears a red hat and carries a tiny backpack full of gadgets. talking to the squirrel again in the magical forest.", "Harold img is a curious and clever boy with bright blue eyes and messy brown hair. He always wears a red hat and carries a tiny backpack full of gadgets. telling his grandmother about his adventure at her home."],
      "negative_prompt": "naked, deformed, bad anatomy, disfigured, poorly drawn face, mutation, extra limb, ugly, disgusting, poorly drawn hands, missing limb, floating limbs, disconnected limbs, blurry, watermarks, oversaturated, distorted hands, amputation",
      "width": 768,
      "height": 768,
      "sa32": 0.5,
      "sa64": 0.5,
      "guidance_scale": 5.0,
      "num_inference_steps": 25,
      "seed": 42,
      "image_ref": "https://alpinabook.ru/upload/resize_cache/iblock/8d9/550_800_1/8d9cd63476f15e85f0d8796555ab1e6b.jpg"
  }
}
```

## Related Resources

This project is based on original implementation of [StoryDiffusion](https://github.com/HVision-NKU/StoryDiffusion).
