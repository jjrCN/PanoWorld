"""Public manifest dataset for PanoWorld 2D Generator.

The released checkpoint was trained with three conditioning panoramas.  The
legacy training variable names accidentally swapped the first two semantic
roles.  This module uses paper-aligned public names while preserving the exact
model order used during training:

    [visual_memory, geometric_proxy, style_reference]
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


CONDITION_IMAGE_AREA = 384 * 384
VAE_IMAGE_AREA = 1024 * 1024
REQUIRED_FIELDS = ("target_panorama", "geometric_proxy", "visual_memory", "style_reference")
FIELD_ALIASES = {
    "target_panorama": ("target",),
    "geometric_proxy": ("geometry_control",),
    "visual_memory": ("coarse_view",),
}


def circular_pad_image(image: Image.Image, pad_ratio: float = 0.125) -> Tuple[Image.Image, int, int]:
    """Wrap a panorama horizontally by copying both edge regions."""

    if not 0 <= pad_ratio < 0.5:
        raise ValueError(f"pad_ratio must be in [0, 0.5), got {pad_ratio}")
    image = image.convert("RGB")
    width, height = image.size
    pad_width = int(width * pad_ratio)
    if pad_width == 0:
        return image.copy(), width, 0

    padded = Image.new("RGB", (width + 2 * pad_width, height))
    padded.paste(image.crop((width - pad_width, 0, width, height)), (0, 0))
    padded.paste(image, (pad_width, 0))
    padded.paste(image.crop((0, 0, pad_width, height)), (width + pad_width, 0))
    return padded, width, pad_width


def crop_circular_padding(image: Image.Image, pad_ratio: float = 0.125) -> Image.Image:
    """Remove padding produced by :func:`circular_pad_image`."""

    padded_width, height = image.size
    original_width = round(padded_width / (1 + 2 * pad_ratio))
    pad_width = (padded_width - original_width) // 2
    return image.crop((pad_width, 0, pad_width + original_width, height))


def area_preserving_size(area: int, aspect_ratio: float) -> Tuple[int, int]:
    """Return a width/height pair with dimensions divisible by 32."""

    if area <= 0 or aspect_ratio <= 0:
        raise ValueError("area and aspect_ratio must be positive")
    raw_width = math.sqrt(area * aspect_ratio)
    raw_height = raw_width / aspect_ratio
    width = round(raw_width / 32) * 32
    height = round(raw_height / 32) * 32
    return max(width, 32), max(height, 32)


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Read and validate a PanoWorld JSONL manifest."""

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
            for field, aliases in FIELD_ALIASES.items():
                if not record.get(field):
                    for alias in aliases:
                        if record.get(alias):
                            record[field] = record[alias]
                            break
            missing = [field for field in REQUIRED_FIELDS if not record.get(field)]
            if missing:
                raise ValueError(f"Missing {missing} at {manifest_path}:{line_number}")
            record.setdefault("id", f"sample-{line_number:06d}")
            record.setdefault("prompt", "")
            records.append(record)
    if not records:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    ids = [str(record["id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Manifest sample IDs must be unique")
    return records


def resolve_record_paths(
    records: Iterable[Dict[str, Any]], manifest_path: str | Path, data_root: Optional[str | Path] = None
) -> List[Dict[str, Any]]:
    """Resolve image paths without mutating the caller's records."""

    base = Path(data_root).expanduser().resolve() if data_root else Path(manifest_path).expanduser().resolve().parent
    resolved: List[Dict[str, Any]] = []
    for record in records:
        item = dict(record)
        for field in REQUIRED_FIELDS:
            value = Path(str(item[field])).expanduser()
            item[field] = str(value if value.is_absolute() else (base / value).resolve())
        resolved.append(item)
    return resolved


class PanoWorldTrainingDataset(Dataset):
    """Dataset that reproduces the released three-image conditioning layout."""

    def __init__(
        self,
        manifest: str | Path,
        data_root: Optional[str | Path] = None,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
        inputs_are_padded: bool = True,
        pad_ratio: float = 0.125,
        style_image_aug_config: Optional[Dict[str, float]] = None,
        curriculum_learning: bool = False,
    ) -> None:
        self.manifest = str(Path(manifest).expanduser().resolve())
        self.data = resolve_record_paths(read_jsonl(self.manifest), self.manifest, data_root)
        self.target_width = target_width
        self.target_height = target_height
        self.inputs_are_padded = inputs_are_padded
        self.pad_ratio = pad_ratio
        self.style_image_aug_config = style_image_aug_config or {}
        self.curriculum_learning = curriculum_learning
        self.current_training_stage = 1
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)]
        )

        if (target_width is None) != (target_height is None):
            raise ValueError("target_width and target_height must be provided together")
        if target_width is not None and (target_width % 32 or target_height % 32):
            raise ValueError("target dimensions must be divisible by 32")

        for item in self.data:
            with Image.open(item["target_panorama"]) as image:
                width, height = image.size
            if not inputs_are_padded:
                width += 2 * int(width * pad_ratio)
            if target_width is not None:
                width, height = target_width, target_height
            item["bucket_resolution"] = (width, height)

    def set_training_stage(self, stage: int) -> None:
        if stage not in (1, 2):
            raise ValueError(f"stage must be 1 or 2, got {stage}")
        self.current_training_stage = stage

    @staticmethod
    def _open(path: str) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB")

    def _resize_inputs(
        self,
        target_panorama: Image.Image,
        geometric_proxy: Image.Image,
        visual_memory: Image.Image,
        style_reference: Image.Image,
    ) -> Tuple[Image.Image, Image.Image, Image.Image, Image.Image]:
        if not self.inputs_are_padded:
            target_panorama, _, _ = circular_pad_image(target_panorama, self.pad_ratio)
            geometric_proxy, _, _ = circular_pad_image(geometric_proxy, self.pad_ratio)
            visual_memory, _, _ = circular_pad_image(visual_memory, self.pad_ratio)

        if self.target_width is not None:
            padded_size = (self.target_width, self.target_height)
            target_panorama = target_panorama.resize(padded_size, Image.Resampling.LANCZOS)
            geometric_proxy = geometric_proxy.resize(padded_size, Image.Resampling.NEAREST)
            visual_memory = visual_memory.resize(padded_size, Image.Resampling.LANCZOS)
            original_width = round(self.target_width / (1 + 2 * self.pad_ratio))
            style_reference = style_reference.resize((original_width, self.target_height), Image.Resampling.LANCZOS)
        return target_panorama, geometric_proxy, visual_memory, style_reference

    def _augment_style(self, image: Image.Image) -> Image.Image:
        flip_prob = float(self.style_image_aug_config.get("random_flip_prob", 0.0))
        rotation_prob = float(self.style_image_aug_config.get("random_rotation_prob", 0.0))
        max_angle = float(self.style_image_aug_config.get("max_rotation_angle", 0.0))
        if flip_prob and random.random() < flip_prob:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if rotation_prob and max_angle and random.random() < rotation_prob:
            image = image.rotate(
                random.uniform(-max_angle, max_angle),
                resample=Image.Resampling.BILINEAR,
                fillcolor=(128, 128, 128),
            )
        return image

    def get_example(self, index: int) -> Dict[str, Any]:
        item = self.data[index]
        target_panorama = self._open(item["target_panorama"])
        geometric_proxy = self._open(item["geometric_proxy"])
        visual_memory = self._open(item["visual_memory"])
        style_reference = self._open(item["style_reference"])
        target_panorama, geometric_proxy, visual_memory, style_reference = self._resize_inputs(
            target_panorama, geometric_proxy, visual_memory, style_reference
        )

        target_width, target_height = target_panorama.size
        visual_memory_cond_size = area_preserving_size(
            CONDITION_IMAGE_AREA, visual_memory.width / visual_memory.height
        )
        visual_memory_vae_size = area_preserving_size(VAE_IMAGE_AREA, visual_memory.width / visual_memory.height)
        geometric_proxy_cond_size = area_preserving_size(
            CONDITION_IMAGE_AREA, geometric_proxy.width / geometric_proxy.height
        )
        geometric_proxy_vae_size = area_preserving_size(VAE_IMAGE_AREA, geometric_proxy.width / geometric_proxy.height)
        style_cond_size = area_preserving_size(CONDITION_IMAGE_AREA, style_reference.width / style_reference.height)
        style_vae_size = area_preserving_size(VAE_IMAGE_AREA, style_reference.width / style_reference.height)

        visual_memory_cond = visual_memory.resize(visual_memory_cond_size, Image.Resampling.LANCZOS)
        geometric_proxy_cond = geometric_proxy.resize(geometric_proxy_cond_size, Image.Resampling.NEAREST)
        style_cond = style_reference.resize(style_cond_size, Image.Resampling.LANCZOS)
        visual_memory_vae = visual_memory.resize(visual_memory_vae_size, Image.Resampling.LANCZOS)
        geometric_proxy_vae = geometric_proxy.resize(geometric_proxy_vae_size, Image.Resampling.NEAREST)
        style_vae = self._augment_style(style_reference.resize(style_vae_size, Image.Resampling.LANCZOS))

        return {
            "sample_id": str(item["id"]),
            "captions": str(item.get("prompt", "")),
            "pixel_values": self.transform(target_panorama),
            "visual_memory_cond_pil": visual_memory_cond,
            "geometric_proxy_cond_pil": geometric_proxy_cond,
            "style_img_cond_pil": style_cond,
            "visual_memory_vae": self.transform(visual_memory_vae),
            "geometric_proxy_vae": self.transform(geometric_proxy_vae),
            "style_img_vae": self.transform(style_vae),
            "bucket_resolution": (target_width, target_height),
            "visual_memory_cond_size": visual_memory_cond_size,
            "visual_memory_vae_size": visual_memory_vae_size,
            "geometric_proxy_cond_size": geometric_proxy_cond_size,
            "geometric_proxy_vae_size": geometric_proxy_vae_size,
            "style_cond_size": style_cond_size,
            "style_vae_size": style_vae_size,
        }

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.get_example(index)

    def __len__(self) -> int:
        return len(self.data)


class BucketDataset(Dataset):
    """Flatten same-resolution batches so each DataLoader batch stays bucket-local."""

    def __init__(self, base_dataset: PanoWorldTrainingDataset, batch_size: int, shuffle: bool = True) -> None:
        self.base_dataset = base_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buckets: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for index, item in enumerate(base_dataset.data):
            self.buckets[item["bucket_resolution"]].append(index)
        self.batch_indices: List[List[int]] = []
        self.flat_indices: List[int] = []
        self.reshuffle()

    def __len__(self) -> int:
        return len(self.flat_indices)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.base_dataset[self.flat_indices[index]]

    def reshuffle(self) -> None:
        self.batch_indices = []
        for _bucket_key, indices in self.buckets.items():
            indices = list(indices)
            if self.shuffle:
                random.shuffle(indices)
            for offset in range(0, len(indices), self.batch_size):
                batch = indices[offset : offset + self.batch_size]
                if len(batch) == self.batch_size:
                    self.batch_indices.append(batch)

        if self.shuffle:
            random.shuffle(self.batch_indices)
        self.flat_indices = [index for batch in self.batch_indices for index in batch]


class BucketSampler:
    """Yield same-resolution index batches for callers that prefer an explicit sampler."""

    def __init__(self, data_source: Sequence[Dict[str, Any]], batch_size: int, shuffle: bool = True) -> None:
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buckets: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for index, item in enumerate(data_source):
            self.buckets[item["bucket_resolution"]].append(index)
        self.bucket_batches = {
            bucket_key: math.ceil(len(indices) / batch_size) for bucket_key, indices in self.buckets.items()
        }

    def __iter__(self):
        batches = []
        for _bucket_key, indices in self.buckets.items():
            indices = list(indices)
            if self.shuffle:
                random.shuffle(indices)
            for offset in range(0, len(indices), self.batch_size):
                batches.append(indices[offset : offset + self.batch_size])

        if self.shuffle:
            random.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return sum(self.bucket_batches.values())


def collate_training_batch(examples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a batch while preserving variable-length PIL prompt inputs."""

    pixel_values = torch.stack([item["pixel_values"] for item in examples])
    if pixel_values.ndim == 4:
        pixel_values = pixel_values.unsqueeze(2)
    visual_memory_vae = torch.stack([item["visual_memory_vae"] for item in examples])
    if visual_memory_vae.ndim == 4:
        visual_memory_vae = visual_memory_vae.unsqueeze(2)
    geometric_proxy_vae = torch.stack([item["geometric_proxy_vae"] for item in examples])
    if geometric_proxy_vae.ndim == 4:
        geometric_proxy_vae = geometric_proxy_vae.unsqueeze(2)
    style_vae = torch.stack([item["style_img_vae"] for item in examples])
    if style_vae.ndim == 4:
        style_vae = style_vae.unsqueeze(2)
    first = examples[0]
    return {
        "sample_ids": [item["sample_id"] for item in examples],
        "pixel_values": pixel_values.contiguous().float(),
        "captions": [item["captions"] for item in examples],
        "visual_memory_cond_pil": [item["visual_memory_cond_pil"] for item in examples],
        "geometric_proxy_cond_pil": [item["geometric_proxy_cond_pil"] for item in examples],
        "style_img_cond_pil": [item["style_img_cond_pil"] for item in examples],
        "visual_memory_vae": visual_memory_vae.contiguous().float(),
        "geometric_proxy_vae": geometric_proxy_vae.contiguous().float(),
        "style_img_vae": style_vae.contiguous().float(),
        "bucket_resolution": first["bucket_resolution"],
        "visual_memory_cond_size": first["visual_memory_cond_size"],
        "visual_memory_vae_size": first["visual_memory_vae_size"],
        "geometric_proxy_cond_size": first["geometric_proxy_cond_size"],
        "geometric_proxy_vae_size": first["geometric_proxy_vae_size"],
        "style_cond_size": first["style_cond_size"],
        "style_vae_size": first["style_vae_size"],
    }
