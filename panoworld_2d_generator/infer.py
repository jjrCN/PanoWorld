"""Native Diffusers inference for the PanoWorld 2D Generator.

The checkpoint was trained with the condition order
``[visual_memory, geometric_proxy, style_reference]``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
from PIL import Image

from .control_generation import (
    DEFAULT_CONTROL_MODEL_ROOT,
    ControlGenerationOptions,
    ControlModelPaths,
    SingleImageControlGenerator,
)
from .control_generation import DEFAULT_BASE_PARAMS_CONFIG
from .data import circular_pad_image, crop_circular_padding


DEFAULT_PROMPT = (
    "Generate a new image based on the visual memory of the first image and the geometric proxy "
    "of the second image. Follow the style, shape, material, color, texture of the third image. "
    "Follow the following "
    "description:"
)
DEFAULT_BASE_MODEL = "Qwen/Qwen-Image-Edit-2509"
DEFAULT_LIGHTNING_LORA = "lightx2v/Qwen-Image-Lightning"
DEFAULT_LIGHTNING_WEIGHT = "Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors"
DEFAULT_PANOWORLD_WEIGHT = "pytorch_lora_weights.safetensors"
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_MODEL_ROOT = DEFAULT_REPO_ROOT / "model_ckpt"
DEFAULT_PANOWORLD_LORA = DEFAULT_LOCAL_MODEL_ROOT / DEFAULT_PANOWORLD_WEIGHT
LIGHTNING_V2_SHA256 = "3e43b9796143ef178dba96707a7c4ec6f882b8ca8afa37577186100d11ad67c5"
INFERENCE_REQUIRED_FIELDS = ("visual_memory", "style_reference")
INFERENCE_PATH_FIELDS = (
    "target_panorama",
    "geometric_proxy",
    "white_model_panorama",
    "visual_memory",
    "style_reference",
)
FIELD_ALIASES = {
    "target_panorama": ("target",),
    "geometric_proxy": ("geometry_control",),
    "white_model_panorama": ("white_model", "white_model_path", "white_model_image", "white_panorama"),
    "visual_memory": ("coarse_view",),
}

LIGHTNING_SCHEDULER_CONFIG: Dict[str, Any] = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Validation/inference JSONL manifest.")
    parser.add_argument("--data-root", default=None, help="Optional base directory for relative image paths.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--base-revision", default=None)
    parser.add_argument("--panoworld-lora", default=str(DEFAULT_PANOWORLD_LORA))
    parser.add_argument("--panoworld-weight-name", default=DEFAULT_PANOWORLD_WEIGHT)
    parser.add_argument("--panoworld-revision", default=None)
    parser.add_argument("--panoworld-scale", type=float, default=1.0)
    parser.add_argument("--lightning-lora", default=DEFAULT_LIGHTNING_LORA)
    parser.add_argument("--lightning-weight-name", default=DEFAULT_LIGHTNING_WEIGHT)
    parser.add_argument("--lightning-revision", default=None)
    parser.add_argument("--lightning-scale", type=float, default=1.0)
    parser.add_argument("--disable-lightning", action="store_true")
    parser.add_argument("--num-inference-steps", type=int, default=6)
    parser.add_argument("--true-cfg-scale", type=float, default=1.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--height", type=int, default=None, help="Padded output height; defaults to geometry input.")
    parser.add_argument("--width", type=int, default=None, help="Padded output width; defaults to geometry input.")
    parser.add_argument("--pad-ratio", type=float, default=0.125)
    parser.add_argument("--inputs-unpadded", action="store_true")
    parser.add_argument("--keep-padding", action="store_true", help="Keep the padded 2.5:1 output instead of cropping to 2:1.")
    parser.add_argument("--generated-geometric-proxy-dir", default=None)
    parser.add_argument("--regenerate-geometric-proxy", action="store_true")
    parser.add_argument("--regenerate-control-assets", action="store_true")
    parser.add_argument("--base-params-config", default=DEFAULT_BASE_PARAMS_CONFIG)
    parser.add_argument("--control-seed", type=int, default=0)
    parser.add_argument("--control-width", type=int, default=2048)
    parser.add_argument("--control-height", type=int, default=1024)
    parser.add_argument("--canny-regions", type=int, default=10)
    parser.add_argument("--canny-min-side-ratio", type=float, default=0.1)
    parser.add_argument("--control-device", default=None, help="Device for PanoSAMic/MoGe/Mask2Former; defaults to --device.")
    parser.add_argument("--control-model-root", default=str(DEFAULT_CONTROL_MODEL_ROOT))
    parser.add_argument("--panosamic-root", default=None)
    parser.add_argument("--panosamic-checkpoint", default=None)
    parser.add_argument("--panosamic-config", default=None)
    parser.add_argument("--sam-weights", default=None)
    parser.add_argument("--moge-root", default=None)
    parser.add_argument("--moge-model", default=None)
    parser.add_argument("--mmdet-root", default=None)
    parser.add_argument(
        "--mask2former-model",
        default="mask2former_swin-l-p4-w12-384-in21k_16xb1-lsj-100e_coco-panoptic",
    )
    parser.add_argument("--mask2former-weights", default=None)
    parser.add_argument("--normal-batch-size", type=int, default=12)
    parser.add_argument("--wall-batch-size", type=int, default=4)
    parser.add_argument("--wall-id", type=int, default=131)
    parser.add_argument(
        "--condition-order",
        choices=("training", "semantic"),
        default="training",
        help="training=[visual_memory,geometric_proxy,style]; semantic=[geometric_proxy,visual_memory,style] for A/B diagnosis.",
    )
    parser.add_argument("--sample-id", action="append", default=None, help="Only run selected ID; may be repeated.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-inputs", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args(argv)


def canonical_record_value(record: Dict[str, Any], field: str) -> Any:
    if record.get(field):
        return record[field]
    for alias in FIELD_ALIASES.get(field, ()):
        if record.get(alias):
            return record[alias]
    return None


def read_inference_manifest(path: str | Path) -> List[Dict[str, Any]]:
    manifest_path = Path(path).expanduser().resolve()
    records: List[Dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {manifest_path}:{line_number}: {exc}") from exc
            for field in (*INFERENCE_REQUIRED_FIELDS, "target_panorama", "geometric_proxy", "white_model_panorama"):
                value = canonical_record_value(record, field)
                if value:
                    record[field] = value
            missing = [field for field in INFERENCE_REQUIRED_FIELDS if not record.get(field)]
            if missing:
                raise ValueError(f"Missing {missing} at {manifest_path}:{line_number}")
            has_geometric_proxy = bool(record.get("geometric_proxy"))
            has_white_model = bool(record.get("white_model_panorama"))
            if has_geometric_proxy == has_white_model:
                raise ValueError(
                    f"Exactly one of geometric_proxy or white_model_panorama is required at "
                    f"{manifest_path}:{line_number}"
                )
            record.setdefault("id", f"sample-{line_number:06d}")
            record.setdefault("prompt", "")
            records.append(record)
    if not records:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    ids = [str(record["id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Manifest sample IDs must be unique")
    return records


def resolve_inference_record_paths(
    records: Iterable[Dict[str, Any]], manifest_path: str | Path, data_root: Optional[str | Path] = None
) -> List[Dict[str, Any]]:
    base = Path(data_root).expanduser().resolve() if data_root else Path(manifest_path).expanduser().resolve().parent
    resolved: List[Dict[str, Any]] = []
    for record in records:
        item = dict(record)
        for field in INFERENCE_PATH_FIELDS:
            if not item.get(field):
                continue
            value = Path(str(item[field])).expanduser()
            item[field] = str(value if value.is_absolute() else (base / value).resolve())
        resolved.append(item)
    return resolved


def dtype_from_name(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def sha256(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_weight_source(source: str, weight_name: Optional[str]) -> Tuple[str, Optional[str]]:
    """Convert a local safetensors path into a Diffusers directory+name pair."""

    path = Path(source).expanduser()
    if path.is_file():
        return str(path.resolve().parent), path.name
    return source, weight_name


def load_adapter(
    pipeline: QwenImageEditPlusPipeline,
    source: str,
    weight_name: Optional[str],
    adapter_name: str,
    revision: Optional[str],
    local_files_only: bool,
) -> Dict[str, Any]:
    normalized_source, normalized_name = normalize_weight_source(source, weight_name)
    kwargs: Dict[str, Any] = {"adapter_name": adapter_name, "local_files_only": local_files_only}
    if normalized_name:
        kwargs["weight_name"] = normalized_name
    if revision and not Path(normalized_source).exists():
        kwargs["revision"] = revision
    pipeline.load_lora_weights(normalized_source, **kwargs)
    info: Dict[str, Any] = {
        "source": source,
        "resolved_source": normalized_source,
        "weight_name": normalized_name,
        "revision": revision,
    }
    local_weight = Path(normalized_source) / normalized_name if normalized_name else Path(normalized_source)
    if local_weight.is_file():
        info["sha256"] = sha256(local_weight)
        info["size_bytes"] = local_weight.stat().st_size
    return info


def build_pipeline(args: argparse.Namespace) -> Tuple[QwenImageEditPlusPipeline, Dict[str, Any]]:
    dtype = dtype_from_name(args.precision)
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(LIGHTNING_SCHEDULER_CONFIG)
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        args.base_model,
        revision=args.base_revision,
        torch_dtype=dtype,
        scheduler=scheduler,
        local_files_only=args.local_files_only,
    )
    panoworld = load_adapter(
        pipeline,
        args.panoworld_lora,
        args.panoworld_weight_name,
        "panoworld",
        args.panoworld_revision,
        args.local_files_only,
    )
    adapters = ["panoworld"]
    weights = [args.panoworld_scale]
    lightning = None
    if not args.disable_lightning:
        lightning = load_adapter(
            pipeline,
            args.lightning_lora,
            args.lightning_weight_name,
            "lightning",
            args.lightning_revision,
            args.local_files_only,
        )
        adapters.append("lightning")
        weights.append(args.lightning_scale)
    pipeline.set_adapters(adapters, adapter_weights=weights)

    if args.cpu_offload:
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.to(args.device)
    pipeline.set_progress_bar_config(disable=False)
    return pipeline, {"panoworld_lora": panoworld, "lightning_lora": lightning, "adapters": dict(zip(adapters, weights))}


def load_rgb(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def record_value(record: Dict[str, Any], field: str, *aliases: str) -> Any:
    for key in (field, *aliases):
        value = record.get(key)
        if value:
            return value
    raise KeyError(field)


def generated_control_dir(args: argparse.Namespace, output_dir: Path) -> Path:
    if args.generated_geometric_proxy_dir:
        return Path(args.generated_geometric_proxy_dir).expanduser().resolve()
    return output_dir / "generated_geometric_proxy"


def build_control_model_paths(args: argparse.Namespace) -> ControlModelPaths:
    return ControlModelPaths(
        model_root=Path(args.control_model_root),
        panosamic_root=Path(args.panosamic_root) if args.panosamic_root else None,
        panosamic_checkpoint=Path(args.panosamic_checkpoint) if args.panosamic_checkpoint else None,
        panosamic_config=Path(args.panosamic_config) if args.panosamic_config else None,
        sam_weights=Path(args.sam_weights) if args.sam_weights else None,
        moge_root=Path(args.moge_root) if args.moge_root else None,
        moge_model=Path(args.moge_model) if args.moge_model else None,
        mmdet_root=Path(args.mmdet_root) if args.mmdet_root else None,
        mask2former_weights=Path(args.mask2former_weights) if args.mask2former_weights else None,
        mask2former_model=args.mask2former_model,
    )


def build_control_generation_options(args: argparse.Namespace) -> ControlGenerationOptions:
    return ControlGenerationOptions(
        output_size=(args.control_width, args.control_height),
        device=args.control_device or args.device,
        normal_batch_size=args.normal_batch_size,
        wall_batch_size=args.wall_batch_size,
        wall_id=args.wall_id,
        control_seed=args.control_seed,
        canny_regions=args.canny_regions,
        canny_min_side_ratio=args.canny_min_side_ratio,
        base_params_config=args.base_params_config,
        overwrite_assets=args.regenerate_control_assets,
        overwrite_geometric_proxy=args.regenerate_geometric_proxy,
    )


def ensure_geometric_proxy(
    record: Dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
    control_generator: SingleImageControlGenerator | None,
) -> str:
    if record.get("geometric_proxy"):
        return str(record["geometric_proxy"])

    if record.get("generated_geometric_proxy_dir"):
        sample_dir = Path(record["generated_geometric_proxy_dir"]).expanduser().resolve()
    else:
        sample_dir = generated_control_dir(args, output_dir) / str(record["id"])
    sample_dir.mkdir(parents=True, exist_ok=True)
    proxy_path = sample_dir / "geometric_proxy.png"
    if proxy_path.is_file() and not args.regenerate_geometric_proxy:
        record["geometric_proxy"] = str(proxy_path)
        return str(proxy_path)

    if control_generator is None:
        raise ValueError("white_model_panorama records require control generation to be enabled")
    control_generator.generate(record["white_model_panorama"], sample_dir, proxy_path)
    record["geometric_proxy"] = str(proxy_path)
    return str(proxy_path)


def prepare_conditions(
    record: Dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
    control_generator: SingleImageControlGenerator | None,
) -> Tuple[List[Image.Image], int, int]:
    geometric_proxy = load_rgb(ensure_geometric_proxy(record, args, output_dir, control_generator))
    visual_memory = load_rgb(record_value(record, "visual_memory", "coarse_view"))
    style = load_rgb(record["style_reference"])
    if args.inputs_unpadded:
        geometric_proxy, _, _ = circular_pad_image(geometric_proxy, args.pad_ratio)
        visual_memory, _, _ = circular_pad_image(visual_memory, args.pad_ratio)
    if geometric_proxy.size != visual_memory.size:
        raise ValueError(
            f"geometric_proxy and visual_memory must have identical sizes for {record['id']}: "
            f"{geometric_proxy.size} != {visual_memory.size}"
        )
    width = args.width or geometric_proxy.width
    height = args.height or geometric_proxy.height
    if width % 32 or height % 32:
        raise ValueError(f"Output width/height must be divisible by 32, got {width}x{height}")
    if args.condition_order == "training":
        return [visual_memory, geometric_proxy, style], width, height
    return [geometric_proxy, visual_memory, style], width, height


def select_records(records: Iterable[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    selected_ids = set(args.sample_id or [])
    selected = [record for record in records if not selected_ids or str(record["id"]) in selected_ids]
    if selected_ids:
        found = {str(record["id"]) for record in selected}
        missing = sorted(selected_ids - found)
        if missing:
            raise ValueError(f"Requested sample IDs are missing from the manifest: {missing}")
    return selected[: args.max_samples] if args.max_samples is not None else selected


def save_input_copy(record: Dict[str, Any], output_dir: Path) -> None:
    input_dir = output_dir / "inputs" / str(record["id"])
    input_dir.mkdir(parents=True, exist_ok=True)
    field_aliases = {
        "geometric_proxy": ("geometry_control",),
        "white_model_panorama": ("white_model", "white_model_path", "white_model_image", "white_panorama"),
        "visual_memory": ("coarse_view",),
        "style_reference": (),
        "target_panorama": ("target",),
    }
    for field, aliases in field_aliases.items():
        try:
            image = load_rgb(record_value(record, field, *aliases))
        except KeyError:
            continue
        image.save(input_dir / f"{field}.png")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    records = resolve_inference_record_paths(read_inference_manifest(args.manifest), args.manifest, args.data_root)
    records = select_records(records, args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    control_started = time.time()
    control_records = [record for record in records if record.get("white_model_panorama")]
    if control_records:
        control_generator = SingleImageControlGenerator(build_control_model_paths(args), build_control_generation_options(args))
        try:
            for record in control_records:
                ensure_geometric_proxy(record, args, output_dir, control_generator)
        finally:
            control_generator.close()
    control_seconds = time.time() - control_started

    started = time.time()
    pipeline, model_info = build_pipeline(args)
    load_seconds = time.time() - started
    results: List[Dict[str, Any]] = []

    for index, record in enumerate(records):
        sample_id = str(record["id"])
        output_path = output_dir / f"{sample_id}_output.png"
        if output_path.exists() and not args.overwrite:
            results.append({"id": sample_id, "output": str(output_path), "status": "skipped"})
            continue
        conditions, width, height = prepare_conditions(record, args, output_dir, None)
        prompt = f"{DEFAULT_PROMPT} {str(record.get('prompt', '')).strip()}".strip()
        generator = torch.Generator(device=args.device).manual_seed(args.seed + index)
        if args.save_inputs:
            save_input_copy(record, output_dir)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        item_started = time.time()
        with torch.inference_mode():
            image = pipeline(
                image=conditions,
                prompt=prompt,
                negative_prompt="",
                true_cfg_scale=args.true_cfg_scale,
                guidance_scale=args.guidance_scale,
                height=height,
                width=width,
                num_inference_steps=args.num_inference_steps,
                max_sequence_length=args.max_sequence_length,
                generator=generator,
            ).images[0]
        padded_size = image.size
        if not args.keep_padding:
            image = crop_circular_padding(image, args.pad_ratio)
        image.save(output_path)
        result = {
            "id": sample_id,
            "output": str(output_path),
            "status": "generated",
            "seed": args.seed + index,
            "padded_size": list(padded_size),
            "saved_size": list(image.size),
            "elapsed_seconds": time.time() - item_started,
        }
        if torch.cuda.is_available():
            result["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated()
        results.append(result)

    summary = {
        "base_model": args.base_model,
        "base_revision": args.base_revision,
        "model": model_info,
        "num_inference_steps": args.num_inference_steps,
        "condition_order": args.condition_order,
        "precision": args.precision,
        "control_seconds": control_seconds,
        "load_seconds": load_seconds,
        "total_seconds": time.time() - started,
        "results": results,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
