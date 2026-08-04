# Third-party Notices

The PanoWorld 2D Generator code is intended to be released under Apache-2.0.

The default runtime uses external models and libraries. Their original licenses and model cards remain authoritative.

## Base And Inference Models

- Qwen-Image-Edit-2509: Apache-2.0
  https://huggingface.co/Qwen/Qwen-Image-Edit-2509
- Qwen-Image-Lightning: Apache-2.0
  https://huggingface.co/lightx2v/Qwen-Image-Lightning

## Geometric Proxy Control Networks

The `models/control_models` directory may contain local debug copies of the following third-party assets:

- PanoSAMic code and Stanford2D3DS checkpoint: CC BY-NC-SA 4.0
  https://github.com/dfki-av/PanoSAMic
  https://huggingface.co/dfki-av/PanoSAMic
- Meta SAM ViT-H checkpoint: Apache-2.0
  https://github.com/facebookresearch/segment-anything
- MoGe code and `moge-2-vitl-normal` checkpoint: MIT, with DINOv2 components under Apache-2.0
  https://github.com/microsoft/MoGe
  https://huggingface.co/Ruicheng/moge-2-vitl-normal
- MMDetection and Mask2Former COCO panoptic checkpoint: Apache-2.0 code and OpenMMLab model-zoo checkpoint
  https://github.com/open-mmlab/mmdetection

PanoSAMic has non-commercial and ShareAlike restrictions. Do not redistribute it as part of a permissive/commercial public release without a separate license review or replacement.

## Training Code Provenance

The training implementation follows Hugging Face Diffusers conventions and APIs. Keep the upstream Diffusers license notice when redistributing derived training code:

https://github.com/huggingface/diffusers
