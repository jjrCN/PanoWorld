# Control Models

Default layout expected by `panoworld_2d_generator.control_generation`:

```text
control_models/
  panosamic/
    PanoSAMic/
    stanford2d3ds-vith-rgb-fold1/
      model.safetensors
    config_stanford2d3ds_dv.json
    sam_vit_h_4b8939.pth
  moge/
    MoGe/
    moge-2-vitl-normal/
      model.pt
  mmdetection/
    configs/
    mmdet/
    panopticapi/
    mask2former_swin-l-p4-w12-384-in21k_16xb1-lsj-100e_coco-panoptic.pth
```

The current handoff package includes these assets so a teammate can debug geometric proxy generation without hunting for the old internal paths.

Refresh or rebuild instructions are in:

```text
handoff/control_environment_setup/
```

License notes:

- PanoSAMic code/weights: CC BY-NC-SA 4.0.
- SAM ViT-H: Apache-2.0.
- MoGe: MIT, with DINOv2 components under Apache-2.0.
- MMDetection and Mask2Former code: Apache-2.0; checkpoint from the OpenMMLab model zoo.
