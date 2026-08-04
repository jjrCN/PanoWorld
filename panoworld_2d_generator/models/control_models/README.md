# Control models

The open-source repository contains only these files initially:

```text
control_models/
  README.md
  panorama_normal.py
  prepare_control_models.sh
```

`panorama_normal.py` is PanoWorld's panorama-normal fusion implementation. The downloaded MoGe repository remains unchanged.

## Download

Run the preparation script on a machine with internet access:

```bash
cd panoworld_2d_generator/models/control_models
bash prepare_control_models.sh
```

The script downloads the public repositories and weights used by `control_generation.py`:

- PanoSAMic and its Stanford2D3DS RGB fold-1 checkpoint;
- Meta SAM ViT-H;
- MoGe and `moge-2-vitl-normal`;
- MMDetection v3.3.0, COCO panopticapi, and Mask2Former Swin-L COCO-Panoptic.

After preparation, the directory contains:

```text
control_models/
  panosamic/
  moge/
  mmdetection/
```

The script is resumable and skips repositories and weights that already exist.

## Run

`control_generation.py` uses this directory by default, so no additional model-root argument is required:

```bash
PYTHONPATH=code python -m panoworld_2d_generator.control_generation \
  --white-model-panorama /path/to/panorama.png \
  --output-dir /path/to/output \
  --overwrite-assets \
  --overwrite
```

The preparation script downloads code and weights only. Install the Python, PyTorch, MMCV, MMEngine, OpenCV, and CUDA dependencies separately for your platform.
