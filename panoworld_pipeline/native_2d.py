"""Code-level bridge from LRM memory renders to the PanoWorld 2D Generator."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

import torch

from panoworld_2d_generator.infer import (
    DEFAULT_BASE_MODEL,
    DEFAULT_BASE_PARAMS_CONFIG,
    DEFAULT_CONTROL_MODEL_ROOT,
    DEFAULT_LIGHTNING_LORA,
    DEFAULT_LIGHTNING_WEIGHT,
    DEFAULT_PANOWORLD_LORA,
    DEFAULT_PANOWORLD_WEIGHT,
    DEFAULT_PROMPT,
    build_pipeline,
    build_control_generation_options,
    build_control_model_paths,
    crop_circular_padding,
    ensure_geometric_proxy,
    prepare_conditions,
    SingleImageControlGenerator,
)


def _cfg_get(config, section: str, key: str, default):
    if not hasattr(config, section):
        return default
    values = getattr(config, section)
    if hasattr(values, "get"):
        return values.get(key, default)
    return getattr(values, key, default)


class NativePanoramaGenerator:
    """Run the released 2D Generator directly inside the PanoWorld pipeline."""

    def __init__(self, args: argparse.Namespace, prompt_prefix: str = DEFAULT_PROMPT):
        self.args = args
        self.prompt_prefix = prompt_prefix
        self.repo_root = Path(__file__).resolve().parents[1]
        self.pipeline, self.model_info = build_pipeline(args)
        self.sample_index = 0

    @classmethod
    def from_config(cls, config) -> "NativePanoramaGenerator":
        generator_device = _cfg_get(config, "generator", "device", "cuda")
        if generator_device == "cuda" and not torch.cuda.is_available():
            generator_device = "cpu"

        args = argparse.Namespace(
            base_model=_cfg_get(config, "generator", "base_model", DEFAULT_BASE_MODEL),
            base_revision=_cfg_get(config, "generator", "base_revision", None),
            panoworld_lora=_cfg_get(config, "generator", "panoworld_lora", DEFAULT_PANOWORLD_LORA),
            panoworld_weight_name=_cfg_get(
                config,
                "generator",
                "panoworld_weight_name",
                DEFAULT_PANOWORLD_WEIGHT,
            ),
            panoworld_revision=_cfg_get(config, "generator", "panoworld_revision", None),
            panoworld_scale=float(_cfg_get(config, "generator", "panoworld_scale", 1.0)),
            lightning_lora=_cfg_get(config, "generator", "lightning_lora", DEFAULT_LIGHTNING_LORA),
            lightning_weight_name=_cfg_get(
                config,
                "generator",
                "lightning_weight_name",
                DEFAULT_LIGHTNING_WEIGHT,
            ),
            lightning_revision=_cfg_get(config, "generator", "lightning_revision", None),
            lightning_scale=float(_cfg_get(config, "generator", "lightning_scale", 1.0)),
            disable_lightning=bool(_cfg_get(config, "generator", "disable_lightning", False)),
            num_inference_steps=int(_cfg_get(config, "generator", "num_inference_steps", 6)),
            true_cfg_scale=float(_cfg_get(config, "generator", "true_cfg_scale", 1.0)),
            guidance_scale=float(_cfg_get(config, "generator", "guidance_scale", 1.0)),
            max_sequence_length=int(_cfg_get(config, "generator", "max_sequence_length", 512)),
            seed=int(_cfg_get(config, "generator", "seed", 0)),
            precision=_cfg_get(config, "generator", "precision", "bf16"),
            device=generator_device,
            cpu_offload=bool(_cfg_get(config, "generator", "cpu_offload", False)),
            height=_cfg_get(config, "generator", "height", None),
            width=_cfg_get(config, "generator", "width", None),
            pad_ratio=float(_cfg_get(config, "generator", "pad_ratio", 0.125)),
            inputs_unpadded=bool(_cfg_get(config, "generator", "inputs_unpadded", True)),
            keep_padding=bool(_cfg_get(config, "generator", "keep_padding", False)),
            condition_order=_cfg_get(config, "generator", "condition_order", "training"),
            local_files_only=bool(_cfg_get(config, "generator", "local_files_only", False)),
            generated_geometric_proxy_dir=_cfg_get(config, "generator", "generated_geometric_proxy_dir", None),
            regenerate_geometric_proxy=bool(_cfg_get(config, "generator", "regenerate_geometric_proxy", False)),
            regenerate_control_assets=bool(_cfg_get(config, "generator", "regenerate_control_assets", False)),
            base_params_config=_cfg_get(config, "generator", "base_params_config", DEFAULT_BASE_PARAMS_CONFIG),
            control_seed=int(_cfg_get(config, "generator", "control_seed", 0)),
            control_width=int(_cfg_get(config, "generator", "control_width", 2048)),
            control_height=int(_cfg_get(config, "generator", "control_height", 1024)),
            canny_regions=int(_cfg_get(config, "generator", "canny_regions", 10)),
            canny_min_side_ratio=float(_cfg_get(config, "generator", "canny_min_side_ratio", 0.1)),
            control_device=_cfg_get(config, "generator", "control_device", None),
            control_python_bin=_cfg_get(
                config,
                "generator",
                "control_python_bin",
                os.environ.get("PANOWORLD_CONTROL_PYTHON"),
            ),
            control_model_root=_cfg_get(config, "generator", "control_model_root", str(DEFAULT_CONTROL_MODEL_ROOT)),
            panosamic_root=_cfg_get(config, "generator", "panosamic_root", None),
            panosamic_checkpoint=_cfg_get(config, "generator", "panosamic_checkpoint", None),
            panosamic_config=_cfg_get(config, "generator", "panosamic_config", None),
            sam_weights=_cfg_get(config, "generator", "sam_weights", None),
            moge_root=_cfg_get(config, "generator", "moge_root", None),
            moge_model=_cfg_get(config, "generator", "moge_model", None),
            mmdet_root=_cfg_get(config, "generator", "mmdet_root", None),
            mask2former_model=_cfg_get(
                config,
                "generator",
                "mask2former_model",
                "mask2former_swin-l-p4-w12-384-in21k_16xb1-lsj-100e_coco-panoptic",
            ),
            mask2former_weights=_cfg_get(config, "generator", "mask2former_weights", None),
            normal_batch_size=int(_cfg_get(config, "generator", "normal_batch_size", 12)),
            wall_batch_size=int(_cfg_get(config, "generator", "wall_batch_size", 4)),
            wall_id=int(_cfg_get(config, "generator", "wall_id", 131)),
        )
        prompt_prefix = _cfg_get(config, "generator", "prompt_prefix", DEFAULT_PROMPT)
        return cls(args, prompt_prefix=prompt_prefix)

    def _generated_proxy_path(self, output_dir: Path, sample_id: str) -> Path:
        if self.args.generated_geometric_proxy_dir:
            root = Path(self.args.generated_geometric_proxy_dir).expanduser().resolve()
            return root / sample_id / "geometric_proxy.png"
        return output_dir / "generated_geometric_proxy" / "geometric_proxy.png"

    def _control_python_bin(self) -> Optional[str]:
        if self.args.control_python_bin:
            return str(self.args.control_python_bin)
        default_python = Path("/opt/diffusers/bin/python")
        if default_python.is_file() and default_python.resolve() != Path(sys.executable).resolve():
            return str(default_python)
        return None

    def _run_control_subprocess(self, white_model_panorama: str, output_dir: Path, sample_id: str) -> str:
        proxy_path = self._generated_proxy_path(output_dir, sample_id)
        if proxy_path.is_file() and not self.args.regenerate_geometric_proxy:
            return str(proxy_path)

        sample_dir = proxy_path.parent
        sample_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            self._control_python_bin(),
            "-m",
            "panoworld_2d_generator.control_generation",
            "--white-model-panorama",
            white_model_panorama,
            "--output-dir",
            str(sample_dir),
            "--output-name",
            proxy_path.name,
            "--control-model-root",
            str(self.args.control_model_root),
            "--device",
            str(self.args.control_device or self.args.device),
            "--control-width",
            str(self.args.control_width),
            "--control-height",
            str(self.args.control_height),
            "--normal-batch-size",
            str(self.args.normal_batch_size),
            "--wall-batch-size",
            str(self.args.wall_batch_size),
            "--wall-id",
            str(self.args.wall_id),
            "--control-seed",
            str(self.args.control_seed),
            "--canny-regions",
            str(self.args.canny_regions),
            "--canny-min-side-ratio",
            str(self.args.canny_min_side_ratio),
        ]
        optional_args = {
            "--base-params-config": self.args.base_params_config,
            "--panosamic-root": self.args.panosamic_root,
            "--panosamic-checkpoint": self.args.panosamic_checkpoint,
            "--panosamic-config": self.args.panosamic_config,
            "--sam-weights": self.args.sam_weights,
            "--moge-root": self.args.moge_root,
            "--moge-model": self.args.moge_model,
            "--mmdet-root": self.args.mmdet_root,
            "--mask2former-model": self.args.mask2former_model,
            "--mask2former-weights": self.args.mask2former_weights,
        }
        for flag, value in optional_args.items():
            if value:
                cmd.extend([flag, str(value)])
        if self.args.regenerate_control_assets:
            cmd.append("--overwrite-assets")
        if self.args.regenerate_geometric_proxy:
            cmd.append("--overwrite")

        env = os.environ.copy()
        env["PYTHONPATH"] = f"{self.repo_root}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(self.repo_root)
        subprocess.run(cmd, check=True, cwd=str(self.repo_root), env=env)
        if not proxy_path.is_file():
            raise FileNotFoundError(str(proxy_path))
        return str(proxy_path)

    def _ensure_proxy_from_white_model(self, white_model_panorama: str, output_dir: Path, sample_id: str) -> str:
        if self._control_python_bin():
            return self._run_control_subprocess(white_model_panorama, output_dir, sample_id)

        record = {"id": sample_id, "white_model_panorama": white_model_panorama}
        if not self.args.generated_geometric_proxy_dir:
            record["generated_geometric_proxy_dir"] = str(output_dir / "generated_geometric_proxy")
        control_generator = SingleImageControlGenerator(
            build_control_model_paths(self.args),
            build_control_generation_options(self.args),
        )
        try:
            return ensure_geometric_proxy(record, self.args, output_dir, control_generator)
        finally:
            control_generator.close()

    def generate(
        self,
        *,
        white_model_panorama: str,
        coarse_view: str,
        style_reference: str,
        output_path: str,
        prompt: str = "",
        extra_output_paths: Optional[Iterable[str]] = None,
    ) -> str:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        sample_id = output.stem
        geometric_proxy = self._ensure_proxy_from_white_model(white_model_panorama, output.parent, sample_id)
        record = {
            "id": sample_id,
            "geometric_proxy": geometric_proxy,
            "visual_memory": coarse_view,
            "style_reference": style_reference,
            "prompt": prompt,
        }
        conditions, width, height = prepare_conditions(record, self.args, output.parent, None)
        full_prompt = f"{self.prompt_prefix} {prompt}".strip()
        generator = torch.Generator(device=self.args.device).manual_seed(self.args.seed + self.sample_index)
        self.sample_index += 1

        with torch.inference_mode():
            image = self.pipeline(
                image=conditions,
                prompt=full_prompt,
                negative_prompt="",
                true_cfg_scale=self.args.true_cfg_scale,
                guidance_scale=self.args.guidance_scale,
                height=height,
                width=width,
                num_inference_steps=self.args.num_inference_steps,
                max_sequence_length=self.args.max_sequence_length,
                generator=generator,
            ).images[0]

        if not self.args.keep_padding:
            image = crop_circular_padding(image, self.args.pad_ratio)

        image.save(output)

        for alias_path in extra_output_paths or []:
            alias = Path(alias_path)
            alias.parent.mkdir(parents=True, exist_ok=True)
            if alias.resolve() != output.resolve():
                shutil.copyfile(output, alias)
        return str(output)
