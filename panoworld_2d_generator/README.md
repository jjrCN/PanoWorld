# PanoWorld 2D Generator

This package contains the open-source 2D generator code for PanoWorld:

- `train.py`: LoRA training for Qwen-Image-Edit-2509.
- `infer.py`: Diffusers inference with three panorama conditions.
- `control_generation.py`: single-image geometric proxy generation from a white-model panorama.
- `data.py` and `model.py`: manifest loading, preprocessing, model loading, optimizer/scheduler setup, and one-step training logic.

The checkpoint-compatible condition order is fixed:

```text
[visual_memory, geometric_proxy, style_reference]
```

Do not swap this order when building manifests or debugging inference.

## Directory Layout

```text
panoworld_2d_generator/
  README.md
  THIRD_PARTY_NOTICES.md
  pyproject.toml
  requirements.txt
  train.py
  infer.py
  data.py
  model.py
  control_generation.py
  scripts/
    train.sh
    infer.sh
  models/
    panoworld_2d_generator/pytorch_lora_weights.safetensors
    control_models/
      panosamic/
      moge/
      mmdetection/
```

The demo inference manifest is stored at the repository root as
`data_list/data_demo_data/inference_2d_generator.jsonl`; its sample assets are
stored under `examples/full_pipeline_demo_datas/`.

`models/control_models` contains PanoSAMic, MoGe, and Mask2Former assets for local debugging. Review their upstream licenses before redistributing these files.

## Environment

The validated environment uses Python 3.10, PyTorch 2.6.0, CUDA 12.4, Diffusers 0.36.0, Transformers 4.52.4, Accelerate 1.10.1, PEFT 0.17.1, and bf16.

```bash
python -m pip install -r panoworld_2d_generator/requirements.txt
python -m pip install -e ./panoworld_2d_generator
```

For H200 control generation, Mask2Former requires a full MMCV build with sm90 CUDA kernels. The prepared wheel and setup notes are in:

```text
handoff/control_environment_setup/
```

## Manifest Format

Training and inference consume JSON Lines. Relative paths are resolved from the manifest directory unless `--data-root` is provided.

```json
{"id":"scene_0001","target_panorama":"images/target.png","geometric_proxy":"images/geometric_proxy.png","visual_memory":"images/visual_memory.png","style_reference":"images/style_reference.png","prompt":"","inputs_are_padded":false}
```

Required fields:

- `target_panorama`
- `geometric_proxy`
- `visual_memory`
- `style_reference`

The loader still accepts legacy aliases `target`, `geometry_control`, and `coarse_view` for old manifests, but new data should use the names above.

Use `--inputs-unpadded` for raw `2048x1024` 2:1 inputs. Old padded validation manifests use `2560x1024` target/proxy/memory images and should run without `--inputs-unpadded`.

## Generate A Geometric Proxy

If `geometric_proxy` is not already prepared, generate it from one white-model panorama:

```bash
python -m panoworld_2d_generator.control_generation \
  --white-model-panorama /path/to/panoImage_2048.png \
  --output-dir /path/to/output_dir \
  --device cuda:0
```

This writes:

```text
segmentation.png
surface_normal.png
wall_mask.png
geometric_proxy.png
```

The equivalent installed command is:

```bash
panoworld-2d-geometric-proxy --white-model-panorama /path/to/pano.png --output-dir /path/to/output
```

The synthesis order is segmentation, wall fill, normal fill, then red Canny edges.

## Inference

```bash
bash scripts/infer_2d_generator.sh
```

By default, the script reads `data_list/data_demo_data/inference_2d_generator.jsonl`,
writes to `./outputs/2d_generator_demo`, and enables `--inputs-unpadded` for
the released 2:1 demo panoramas. Set `MANIFEST`, `OUTPUT_DIR`, or
`INPUTS_UNPADDED=0` to override these defaults.

The inference model receives exactly three image conditions in the training order:
`visual_memory`, `geometric_proxy`, and `style_reference`.  White-model panoramas
must first be converted into geometric proxies with `control_generation.py`; they
are not fed directly to Qwen-Image.

Default local checkpoints are resolved from the repository root:

```text
model_ckpt/Qwen-Image-Edit-2509
model_ckpt/pytorch_lora_weights.safetensors
model_ckpt/Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors
```

Set `BASE_MODEL`, `PANOWORLD_LORA`, `LIGHTNING_LORA`, `CONTROL_MODEL_ROOT`,
`NUM_INFERENCE_STEPS`, or `PRECISION` to override the default inference setup.
Less common options can be appended directly to the script command and are
forwarded to `panoworld_2d_generator.infer`.

## Training

```bash
export TRAIN_MANIFEST=/path/to/train_manifest.jsonl
export DATA_ROOT=/path/to/processed_training_data
export OUTPUT_DIR=/path/to/train_outputs
bash panoworld_2d_generator/scripts/train.sh
```

Relative image paths in the manifest are resolved from the manifest directory
unless `DATA_ROOT` or `TRAIN_DATA_ROOT` is provided. The released Front3D
training manifest stores scene-relative paths, so `DATA_ROOT` should point to
the processed Front3D training-data root. The script defaults to a single-node
launch and can be scaled with standard torchrun variables: `NUM_GPUS`,
`NNODES`, `NODE_RANK`, `MASTER_ADDR`, and `MASTER_PORT`.

The default recipe uses bf16, per-device batch size 1, gradient accumulation 4,
LoRA rank/alpha 64, learning rate `1e-4`, cosine schedule, warmup 150, and
5000 steps. Common overrides are exposed as environment variables:
`MIXED_PRECISION`, `TRAIN_BATCH_SIZE`, `GRAD_ACCUM_STEPS`, `LEARNING_RATE`,
`MAX_TRAIN_STEPS`, `NUM_WORKERS`, `CHECKPOINTING_STEPS`, `LORA_RANK`,
`LORA_ALPHA`, `REPORT_TO`, and `GRADIENT_CHECKPOINTING`. The script passes
`--inputs_unpadded` by default for the new 2:1 panorama training data.

For old padded data, append:

```bash
--inputs_padded
```

## Handoff

Operational notes, control-environment setup, MMCV wheel provenance, and cleanup decisions are kept outside the code package under:

```text
handoff/
```
