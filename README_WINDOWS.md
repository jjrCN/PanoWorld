# PanoWorld on Windows (RTX 50xx / Blackwell)

The upstream `requirements.txt` targets Linux + CUDA 12.4 + sm80/sm90. This checkout was set up on
Windows 11 with an RTX 5080 Laptop GPU (sm_120), which needs CUDA 12.8+ and a source build of gsplat.

## What was installed

| Component | Location / version |
| --- | --- |
| Python | 3.10.22 (uv-managed, installed into `.python/` inside the repo), venv at `.venv` |
| PyTorch | 2.7.0+cu128, torchvision 0.22.0+cu128, xformers 0.0.30 |
| Other Python deps | `requirements-windows.txt` (same pins as upstream, minus Linux-only wheels) |
| gsplat | 1.5.3 (pure-Python wheel; CUDA kernels JIT-compiled for sm_120 on first import, cached in `.torch_extensions/`) |
| CUDA Toolkit | 12.8 at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8` |
| MSVC | Visual Studio 2022 Build Tools, C++ workload |
| Checkpoints | `model_ckpt/` (LRM 1024x512, LRM 2048x1024, PanoWorld LoRA, Lightning LoRA) |
| Eval data | `data/RealSee3D_eval_data/RealSee3D_eval_data/` (RealSee3D eval set from HF) |

## Running

Every command needs MSVC + CUDA on PATH (gsplat checks its JIT cache at import). Use the wrapper:

```bat
scripts\env_windows.cmd python -m panoworld_lrm.inference --config configs/inference_1024_512_windows.yaml
```

The Bash scripts under `scripts/` also work from Git Bash (`scripts/env.sh` was patched to find
`.venv/Scripts/python.exe`), but run them from a shell where `vcvars64.bat` has been called.

`configs/inference_1024_512_windows.yaml` limits input views to 3 (`viewpoint_max_view`) so the
1024x512 LRM fits in 16 GB. Raise it on larger GPUs. The 2048x1024 LRM needs ~19 GB for a single
view and will not fit on this GPU.

## Recreating the venv

```bat
uv python install 3.10 --install-dir .python
uv venv .venv --python .python\cpython-3.10.22-windows-x86_64-none\python.exe
uv pip install --python .venv\Scripts\python.exe --index-url https://download.pytorch.org/whl/cu128 torch==2.7.0+cu128 torchvision==0.22.0+cu128 xformers==0.0.30
uv pip install --python .venv\Scripts\python.exe -r requirements-windows.txt
.venv\Scripts\python.exe scripts\patch_gsplat_windows.py
```

## Not set up

- **mmcv / mmdet** (control models for the 2D Generator and full pipeline): the bundled
  `wheelhouse/mmcv-*.whl` is Linux-only and no Windows wheel exists for torch 2.7 / cu128.
  Build from source (`pip install mmcv==2.2.0 --no-binary mmcv` inside `env_windows.cmd`) if needed.
- **Qwen-Image-Edit-2509 base model** (~20 GB) was not downloaded. The 2D Generator (PanoWorld-DiT)
  needs ~47 GB VRAM per the README, so it will not run on this GPU without CPU offloading.

## Re-applying the gsplat patch

If gsplat is reinstalled, run `scripts\patch_gsplat_windows.py` again (it swaps GCC-only host
compiler flags for MSVC ones), then delete
`.torch_extensions\gsplat_cuda` to force a rebuild.
