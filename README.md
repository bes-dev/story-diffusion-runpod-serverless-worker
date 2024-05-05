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

## Continuous Deployment
This worker follows a modified version of the [worker template](https://github.com/runpod-workers/worker-template) where the Docker build workflow contains additional SD models to be built and pushed.

## API

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
  }
}
```
