INPUT_SCHEMA = {
    "prompts": {
        "type": list,
        "required": True
    },
    "negative_prompt": {
        "type": str,
        "required": True
    },
    "width": {
        "type": int,
        "required": True
    },
    "height": {
        "type": int,
        "required": True
    },
    "sa32": {
        "type": float,
        "required": True
    },
    "sa64": {
        "type": float,
        "required": True,
    },
    "guidance_scale": {
        "type": float,
        "required": True
    },
    "num_inference_steps": {
        "type": int,
        "required": True
    },
    "seed": {
        "type": int,
        "required": True
    },
    "image_ref": {
        "type": str,
        "required": True
    }
}
