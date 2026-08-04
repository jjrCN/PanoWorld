# Local Models

This directory is the default model root used by `infer.py` and `control_generation.py`.

```text
models/
  panoworld_2d_generator/
    pytorch_lora_weights.safetensors
  control_models/
    panosamic/
    moge/
    mmdetection/
```

`panoworld_2d_generator/pytorch_lora_weights.safetensors` is the trained PanoWorld LoRA used by the default inference script.

`control_models/` contains the three auxiliary control networks for debugging single-image white-model input:

- PanoSAMic + SAM ViT-H for Stanford13-style segmentation.
- MoGe for panorama surface normals.
- MMDetection Mask2Former for COCO panoptic wall masks.

These third-party control assets are convenient for internal handoff and debugging. Check upstream licenses before redistributing them in a public release.
