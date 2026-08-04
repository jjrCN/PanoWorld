# PanoWorld: A Generative Spatial World Model for Consistent Whole-House Panorama Synthesis

<p align="center">
  <strong>Jinrang Jia, Zhenjia Li, Yijiang Hu, Yifeng Shi</strong>
</p>

<p align="center">
  <strong>Ke Holdings Inc.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.17916"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2605.17916-b31b1b.svg"></a>
  <a href="https://jjrcn.github.io/PanoWorld-project-home/"><img alt="Project Page" src="https://img.shields.io/badge/Project-Page-2f80ed.svg"></a>
  <a href="https://huggingface.co/JiaJinrang/PanoWorld/tree/main"><img alt="Model" src="https://img.shields.io/badge/Model-HuggingFace-f97316.svg"></a>
  <a href="https://huggingface.co/datasets/JiaJinrang/PanoWorld"><img alt="Dataset" src="https://img.shields.io/badge/Dataset-HuggingFace-10b981.svg"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
</p>

PanoWorld generates consistent whole-house 360-degree panoramas from floorplan-guided viewpoints and a style reference. The pipeline couples high-fidelity 2D panorama generation with an explicit renderable 3DGS memory state reconstructed by PanoWorld-LRM.

<p align="center">
  <img src="assets/panoworld.png" alt="PanoWorld main figure" width="95%">
</p>

## What Is Included

- **PanoWorld-LRM inference**: multi-view panoramic LRM reconstruction and 3DGS export.
- **PanoWorld-LRM training**: DDP training code with image, perceptual, opacity, and position-depth supervision.
- **PanoWorld 2D Generator training/inference**: LoRA training and native Diffusers inference based on Qwen-Image-Edit.
- **Full PanoWorld inference**: progressive multi-node generation using LRM renders plus the 2D Generator in one Python pipeline.

## News

- `Coming soon`: Processed RealSee3D and 3D-FRONT training data will be released in one week.
- `2026-08-04`: Released the PanoWorld-v1.0 codebase, including PanoWorld-LRM training/inference, PanoWorld 2D Generator LoRA training/inference, and end-to-end progressive PanoWorld inference.
- `2026-08-04`: Added one-command scripts, a unified environment, demo manifests under `data_list/`, sample assets under `examples/`, and a lightweight WebGL panorama viewer.
- `2026-08-04`: Released the PanoWorld 2D Generator LoRA checkpoint on Hugging Face under `model_ckpt/pytorch_lora_weights.safetensors`.
- `2026-07-20`: PanoWorld has been **conditionally accepted as a Conference Paper to SIGGRAPH Asia 2026**. 🎉🎉🎉
- `2026-05-25`: Open-sourced the PanoWorld-LRM inference code, `1024x512` and `2048x1024` checkpoints, and RealSee3D evaluation data.
- `2026-05-19`: Paper released and project page launched.

## Installation

```bash
git clone https://github.com/jjrCN/PanoWorld.git
cd PanoWorld
pip install -r requirements.txt
```

The whole repository uses one environment for PanoWorld-LRM and the 2D Generator. The fixed `requirements.txt` has been validated with Python 3.10, CUDA 12.1, and NVIDIA H200/A100-class GPUs.

## Checkpoints

Download released checkpoints from [JiaJinrang/PanoWorld](https://huggingface.co/JiaJinrang/PanoWorld/tree/main) and place them under `model_ckpt/` or override the paths in YAML configs.

| Component | Default Path |
| --- | --- |
| PanoWorld-LRM 1024x512 | `model_ckpt/ckpt_panoworld_lrm_1024_512.pt` |
| PanoWorld-LRM 2048x1024 | `model_ckpt/ckpt_panoworld_lrm_2048_1024.pt` |
| Qwen-Image-Edit-2509 base model | `model_ckpt/Qwen-Image-Edit-2509` |
| PanoWorld 2D Generator LoRA | `model_ckpt/pytorch_lora_weights.safetensors` |
| Qwen-Image-Lightning LoRA | `model_ckpt/Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors` |

All default configs use these relative paths. Users only need to place the files under `model_ckpt/` or create symlinks with the same names.

## One-Command Scripts

### PanoWorld-LRM Inference

```bash
bash scripts/infer_lrm_1024_512.sh
bash scripts/infer_lrm_2048_1024.sh
```

Update `data.root_data_dir`, `data.data_path`, `inference.ckpt_path`, and `inference.out_dir` in the selected config before running.
After inference, the scripts automatically report the mean PSNR, SSIM, and LPIPS over all exported GT/rendered panorama pairs.

### PanoWorld-LRM Training

```bash
NUM_GPUS=8 bash scripts/train_lrm_1024_512.sh
NUM_GPUS=8 bash scripts/train_lrm_2048_1024.sh
```

The corresponding configs are `configs/train_lrm_1024_512.yaml` and `configs/train_lrm_2048_1024.yaml`. Set `data.root_data_dir` to the processed training-data root, or a list of roots, and set `data.data_path` to the matching manifest path, or list of manifest paths. Each line in a manifest is a relative scene entry such as `scene_000001/map.json`, following the same convention as `data_list/data_realsee3d/realsee3D_train.txt`. The panorama depth and depth scale are read from the current scene directory:

```text
<scene>
  map.json
  viewpoints
    <view>
      panoImage_2048.png
      depth_image.png
      depth_scale.txt
      extrinsics.txt
      transforms.json
```

### PanoWorld 2D Generator

```bash
DATA_ROOT=/path/to/front3d_train_data bash scripts/train_2d_generator.sh
bash scripts/infer_2d_generator.sh
```

The default 2D training manifest is `data_list/data_front3d/train_2d_generator.jsonl`. Because the manifest stores paths relative to the processed training-data root, set `DATA_ROOT` or `TRAIN_DATA_ROOT` before launching training. The default 2D inference manifest is `data_list/data_demo_data/inference_2d_generator.jsonl`, and outputs are written to `./outputs/2d_generator_demo`. Set `MANIFEST` and `OUTPUT_DIR` to override inference. The JSONL manifest format is documented in `panoworld_2d_generator/README.md`. The checkpoint-compatible condition order is:

```text
visual_memory, geometric_proxy, style_reference
```

When a white-model panorama is used as geometry control, it is first converted
into a `geometric_proxy`; the raw `place_image.png` is not fed directly to
Qwen-Image.

### Full PanoWorld Multi-Node Inference

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

All released entry points use this same environment. They first check
`PANOWORLD_PYTHON`, then `${repo}/.venv/bin/python`, and finally the current
shell `python`.

```bash
bash scripts/infer_panoworld.sh
```

`configs/inference_panoworld.yaml` enables `data.panoworld_mode=true` and directly connects LRM inference with the native 2D Generator. For each target node, the full pipeline prepares three image conditions for the 2D Generator:

You can switch between the three provided target styles by setting `data.panoworld_start_image` to `panoImage_2048_franch.png`, `panoImage_2048_simple.png`, or `panoImage_2048_chinese.png`. You can also use other image-to-image models to create additional start panoramas in new styles and use them as the first image for subsequent node generation.

- `visual_memory`: the masked LRM memory render at the target viewpoint, used as cross-node appearance and layout memory.
- `geometric_proxy`: the geometric control image converted from the shell-rendered `place_image.png`, used to constrain room structure and furniture layout.
- `style_reference`: the nearest completed panorama or the start panorama, used to transfer the target visual style.

The full-pipeline scene format follows the LRM format and additionally requires `place_image.png`, `place_depth.png`, and `place_depth_scale.txt` under each viewpoint directory.

### Panorama Viewer

```bash
bash scripts/visualize_panoworld.sh examples/full_pipeline_demo_datas/scene0000/viewpoints
```

The viewer starts a lightweight WebGL service for generated panoramas in a `viewpoints` directory. By default it binds to `0.0.0.0:8003`, enumerates the server's reachable hostnames/IP addresses, and prints browser URLs such as `http://<server-ip>:8003/`.

```bash
PORT=8003 bash scripts/visualize_panoworld.sh /path/to/viewpoints
```

If your platform provides a public hostname or proxy address, pass it explicitly so the printed URL is exact:

```bash
PUBLIC_HOST=my-server.example.com bash scripts/visualize_panoworld.sh /path/to/viewpoints
```

## Inference Cost

We report inference memory and runtime on a single NVIDIA H200 GPU, averaged over 50 runs.

| Module | Views | Resolution | Memory | Time |
| --- | ---: | --- | ---: | ---: |
| PanoWorld-LRM | 1 | 1024x512 | 6143 MiB | 0.17s |
| PanoWorld-LRM | 1 | 2048x1024 | 18823 MiB | 2.30s |
| PanoWorld-LRM | 8 | 1024x512 | 27507 MiB | 1.45s |
| PanoWorld-LRM | 8 | 2048x1024 | 108369 MiB | 20.53s |
| PanoWorld-LRM | 12 | 1024x512 | 40285 MiB | 2.28s |
| PanoWorld-LRM | 12 | 2048x1024 | OOM | OOM |
| PanoWorld-DiT | - | 1024x512 | 46742 MiB | 11.00s |

## Data

| Data | Usage | Link |
| --- | --- | --- |
| 3D-FRONT | LRM and 2D Generator training | [Download](https://tianchi.aliyun.com/dataset/65347) |
| RealSee3D | LRM training/evaluation | [Download](https://github.com/realsee-developer/RealSee3D) |
| PanoWorld evaluation assets | LRM evaluation examples | [Hugging Face Dataset](https://huggingface.co/datasets/JiaJinrang/PanoWorld) |

Example manifest templates are provided in `examples/`.

## Output

- **LRM inference** writes rendered target views, depth maps, and an `output_ply` directory under `inference.out_dir`.
- **Full PanoWorld inference** writes intermediate LRM memory renders and final generated panoramas back to the original viewpoint directories. Its multi-node `output_ply` directories are written under `inference.out_dir` and can be opened with SIBR Viewer or SuperSplat.

## Citation

```bibtex
@misc{jia2026panoworldgenerativespatialworld,
      title={PanoWorld: A Generative Spatial World Model for Consistent Whole-House Panorama Synthesis},
      author={Jinrang Jia and Zhenjia Li and Yijiang Hu and Yifeng Shi},
      year={2026},
      eprint={2605.17916},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.17916},
}
```

## License

This project is released under the Apache 2.0 License. Third-party code included in this repository keeps its original license notices.

## Acknowledgements

We thank [QwenLM/Qwen-Image](https://github.com/QwenLM/Qwen-Image), [MVP](https://github.com/Gynjn/MVP), [RealSee3D](https://github.com/realsee-developer/RealSee3D), and [3D-FRONT](https://tianchi.aliyun.com/dataset/65347) for their open-source contributions.
