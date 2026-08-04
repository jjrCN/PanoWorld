"""Single-image geometric proxy generation from white-model panoramas.

This module is intentionally import-light.  Heavy dependencies such as
PanoSAMic, MoGe, and MMDetection are imported only when a white-model panorama
actually needs to be converted into a geometric proxy.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from PIL import Image

np = None
cv2 = None

DEFAULT_BASE_PARAMS_CONFIG = None

BUILTIN_BASE_PARAMS_CONFIG = {
    "ls_l3_core_item_v4": {
        1: [4001, [0, 0, 255], 1],
        2: [4002, [0, 255, 0], 1],
        3: [400206, [0, 255, 255], 1],
        4: [4015, [0, 0, 204], 1],
        5: [4008, [0, 204, 0], 1],
        6: [5004, [0, 204, 204], 1],
        7: [5007, [0, 51, 0], 1],
        8: [5005, [0, 0, 51], 1],
        9: [5006, [0, 51, 51], 1],
        10: [5008, [0, 51, 102], 1],
        11: [4021, [0, 102, 102], 1],
        12: [4017, [0, 102, 51], 1],
        13: [4016, [0, 102, 153], 1],
        14: [4013, [0, 153, 102], 1],
        15: [4014, [0, 153, 153], 0],
        16: [4018, [0, 153, 204], 0],
        17: [4012, [0, 204, 153], 1],
        18: [4010, [0, 204, 255], 1],
        19: [4011, [0, 255, 204], 1],
        20: [5003, [0, 51, 153], 0],
        21: [4020, [0, 153, 51], 1],
        22: [501102, [0, 51, 204], 0],
        23: [5002, [0, 204, 51], 0],
        24: [5010, [0, 51, 255], 0],
        25: [5009, [0, 255, 51], 0],
        26: [501103, [0, 102, 0], 1],
        27: [501204, [0, 0, 102], 0],
        28: [502912, [0, 102, 255], 1],
        29: [502901, [0, 255, 102], 1],
        30: [502908, [0, 153, 0], 1],
        31: [502915, [0, 0, 153], 1],
        32: [501101, [0, 153, 255], 0],
        33: [5012, [0, 255, 153], 1],
        34: [4009, [0, 204, 102], 1],
        35: [502902, [0, 255, 102], 1],
        36: [5014, [0, 255, 0], 1],
    },
    "ls_l3_core_item_no_convex_v4": [
        4001,
        4002,
        400206,
        4015,
        4008,
        5004,
        5007,
        5005,
        5006,
        5008,
        4021,
        502912,
        502901,
        502902,
        502908,
        502915,
        501101,
        5012,
        5014,
    ],
}

WALL_RGB = [0, 102, 204]
EDGE_RGB = [255, 0, 0]


@dataclass(frozen=True)
class ControlAssetPaths:
    panorama: Path
    segmentation: Path
    surface_normal: Path
    wall_mask: Path


@dataclass(frozen=True)
class ControlSynthesisOptions:
    output_size: tuple[int, int] = (2048, 1024)
    canny_regions: int = 10
    canny_min_side_ratio: float = 0.1


def ensure_cv2() -> Any:
    global cv2
    if cv2 is None:
        import cv2 as cv2_module

        cv2 = cv2_module
    return cv2


def ensure_np() -> Any:
    global np
    if np is None:
        import numpy as np_module

        np = np_module
    return np


def load_base_params_config(path: str | Path | None) -> dict[str, Any]:
    if not path and not DEFAULT_BASE_PARAMS_CONFIG:
        return BUILTIN_BASE_PARAMS_CONFIG
    config_path = Path(path or DEFAULT_BASE_PARAMS_CONFIG)
    if config_path.is_file():
        try:
            import yaml

            with config_path.open("r", encoding="utf-8") as handle:
                return yaml.safe_load(handle)
        except Exception as exc:
            print(f"[WARN] failed to read {config_path}, using built-in config: {exc}", file=sys.stderr)
    return BUILTIN_BASE_PARAMS_CONFIG


def parse_size(value: str) -> tuple[int, int]:
    if "x" in value:
        width, height = value.lower().split("x", 1)
    elif "," in value:
        width, height = value.split(",", 1)
    else:
        raise ValueError("size must be WIDTHxHEIGHT")
    return int(width), int(height)


def decode_instance_encoded_rgb(encoded_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode packed RGB ids into semantic ids and instance ids."""

    encoded_rgb = encoded_rgb.astype(np.uint32)
    packed_ids = encoded_rgb[:, :, 0] * 2**16 + encoded_rgb[:, :, 1] * 2**8 + encoded_rgb[:, :, 2]
    primary_id = (packed_ids & 0b111000000000000000000000) >> 21
    secondary_id = (packed_ids & 0b000111111110000000000000) >> 13
    tertiary_id = (packed_ids & 0b000000000001111110000000) >> 7
    instance_id = packed_ids & 0b000000000000000001111111

    semantic_id = primary_id * 1000 + secondary_id
    semantic_id[tertiary_id != 0] = (
        primary_id[tertiary_id != 0] * 100000
        + secondary_id[tertiary_id != 0] * 100
        + tertiary_id[tertiary_id != 0]
    )
    return semantic_id, instance_id


def close_and_fill(mask: np.ndarray, fill_value: int, kernel_width: int = 5) -> np.ndarray:
    cv2_lib = ensure_cv2()
    mask = mask.copy().astype(np.uint8)
    mask = cv2_lib.morphologyEx(mask, cv2_lib.MORPH_CLOSE, np.ones((kernel_width, kernel_width), np.uint8))
    contours, _hierarchy = cv2_lib.findContours(mask, cv2_lib.RETR_EXTERNAL, cv2_lib.CHAIN_APPROX_SIMPLE)
    for index in range(len(contours)):
        cv2_lib.drawContours(mask, contours, index, fill_value, -1)
    return mask


def render_configured_objects(encoded_segmentation_rgb: np.ndarray, palette_config: dict[str, Any]) -> np.ndarray:
    """Render configured object instances from packed segmentation RGB."""

    cv2_lib = ensure_cv2()
    ensure_np()
    palette = {int(key): value for key, value in palette_config["ls_l3_core_item_v4"].items()}
    skip_convex_semantics = set(palette_config["ls_l3_core_item_no_convex_v4"])

    semantic_ids, instance_ids = decode_instance_encoded_rgb(encoded_segmentation_rgb)
    semantic_ids_l2 = semantic_ids // 100
    unique_semantics = np.unique(semantic_ids)
    unique_l2_semantics = unique_semantics // 100
    indexed_regions = np.zeros(semantic_ids.shape, np.uint8)

    for palette_index, item in palette.items():
        semantic_target = int(item[0])
        if semantic_target in unique_semantics:
            semantic_mask = semantic_ids == semantic_target
        elif semantic_target in unique_l2_semantics:
            semantic_mask = semantic_ids_l2 == semantic_target
        else:
            continue

        for semantic_value in np.unique(semantic_ids[semantic_mask]):
            exact_semantic_mask = semantic_ids == semantic_value
            for instance_value in np.unique(instance_ids[exact_semantic_mask]):
                instance_mask = (instance_ids == instance_value) & exact_semantic_mask
                region_mask = (semantic_mask & instance_mask).astype(np.uint8) * 255
                region_mask = close_and_fill(region_mask, fill_value=palette_index)
                indexed_regions[region_mask > 0] = region_mask[region_mask > 0]

                if semantic_target in skip_convex_semantics:
                    continue
                contours, _ = cv2_lib.findContours(region_mask, cv2_lib.RETR_TREE, cv2_lib.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                main_contour = max(contours, key=cv2_lib.contourArea)
                (main_x, main_y), _main_radius = cv2_lib.minEnclosingCircle(main_contour)
                nearby_contours = []
                for contour in contours:
                    (x, y), _radius = cv2_lib.minEnclosingCircle(contour)
                    if abs(x - main_x) > encoded_segmentation_rgb.shape[0] or abs(y - main_y) > encoded_segmentation_rgb.shape[0]:
                        continue
                    nearby_contours.append(contour)
                if nearby_contours:
                    hull = cv2_lib.convexHull(np.vstack(nearby_contours))
                    indexed_regions = cv2_lib.drawContours(indexed_regions, [hull], -1, palette_index, -1)

    rendered = np.zeros((*semantic_ids.shape, 3), np.uint8)
    for palette_index in np.unique(indexed_regions):
        if palette_index == 0:
            continue
        rendered[indexed_regions == palette_index] = palette[int(palette_index)][1]
    return rendered


def random_region_suppression_mask(template: np.ndarray, count: int = 10, min_side_ratio: float = 0.1) -> np.ndarray:
    cv2_lib = ensure_cv2()
    ensure_np()
    mask = np.zeros_like(template)
    min_side = min(mask.shape[:2]) * min_side_ratio
    if min_side <= 10:
        return mask
    for _ in range(count):
        radius = np.random.randint(10, int(min_side) + 11)
        elongated_radius = np.random.randint(20, int(min_side) + 11)
        row = np.random.randint(radius, mask.shape[0] - radius)
        col = np.random.randint(radius, mask.shape[1] - radius)
        cv2_lib.circle(mask, (col, row), radius, 1, -1)
        cv2_lib.ellipse(mask, (col, row), (radius, elongated_radius), 0, 0, 360, 1, -1)
    return mask


def extract_edge_mask(
    panorama_rgb: np.ndarray,
    current_control_rgb: np.ndarray,
    palette_config: dict[str, Any],
    region_count: int = 10,
    min_side_ratio: float = 0.1,
) -> np.ndarray:
    cv2_lib = ensure_cv2()
    ensure_np()
    low_threshold = np.random.randint(100, 200)
    high_threshold = low_threshold + np.random.randint(200, 500)
    edges = cv2_lib.Canny(cv2_lib.cvtColor(panorama_rgb, cv2_lib.COLOR_RGB2GRAY), low_threshold, high_threshold)

    edge_allowed = np.ones(shape=panorama_rgb.shape[:2], dtype=np.uint8)
    no_edge_colors = [item[1] for item in palette_config["ls_l3_core_item_v4"].values() if item[2] == 0]
    for color in no_edge_colors:
        edge_allowed[np.all(current_control_rgb == color, axis=-1)] = 0
    edges[edge_allowed == 0] = 0

    suppressed = (
        random_region_suppression_mask(edges, 2 * region_count, 2 * min_side_ratio)
        if region_count > 0
        else np.zeros_like(edges)
    )
    edges[suppressed > 0] = 0
    return edges


def read_rgb_image(path: Path, output_size: tuple[int, int], interpolation: int) -> np.ndarray:
    cv2_lib = ensure_cv2()
    image = cv2_lib.imread(str(path), cv2_lib.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(str(path))
    image = cv2_lib.cvtColor(image, cv2_lib.COLOR_BGR2RGB)
    return cv2_lib.resize(image, output_size, interpolation=interpolation)


def read_binary_mask(path: Path, output_size: tuple[int, int]) -> np.ndarray:
    cv2_lib = ensure_cv2()
    image = cv2_lib.imread(str(path), cv2_lib.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(str(path))
    if image.ndim == 3:
        image = image[:, :, 0]
    return cv2_lib.resize(image, output_size, interpolation=cv2_lib.INTER_NEAREST)


def synthesize_geometry_control(
    assets: ControlAssetPaths,
    palette_config: dict[str, Any],
    options: ControlSynthesisOptions | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Create a 2048x1024 RGB geometry-control image."""

    cv2_lib = ensure_cv2()
    ensure_np()
    options = options or ControlSynthesisOptions()
    panorama_rgb = read_rgb_image(assets.panorama, options.output_size, cv2_lib.INTER_AREA)
    encoded_segmentation_rgb = read_rgb_image(assets.segmentation, options.output_size, cv2_lib.INTER_NEAREST)
    normal_rgb = read_rgb_image(assets.surface_normal, options.output_size, cv2_lib.INTER_NEAREST)
    wall_mask = read_binary_mask(assets.wall_mask, options.output_size)

    control_rgb = render_configured_objects(encoded_segmentation_rgb, palette_config)
    semantic_pixels = int((np.sum(control_rgb, axis=-1) != 0).sum())

    wall_fill = (wall_mask != 0) & (np.sum(control_rgb, axis=-1) == 0)
    control_rgb[wall_fill] = WALL_RGB

    normal_fill = np.sum(control_rgb, axis=-1) == 0
    control_rgb[normal_fill] = normal_rgb[normal_fill]

    edge_mask = extract_edge_mask(
        panorama_rgb,
        control_rgb,
        palette_config,
        region_count=options.canny_regions,
        min_side_ratio=options.canny_min_side_ratio,
    )
    edge_fill = edge_mask != 0
    control_rgb[edge_fill] = EDGE_RGB

    stats = {
        "semantic": semantic_pixels,
        "wall_added": int(wall_fill.sum()),
        "normal_fill": int(normal_fill.sum()),
        "canny_added": int(edge_fill.sum()),
    }
    return control_rgb, stats


def write_rgb_image(path: Path, image_rgb: np.ndarray) -> None:
    cv2_lib = ensure_cv2()
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2_lib.imwrite(str(path), cv2_lib.cvtColor(image_rgb.astype(np.uint8), cv2_lib.COLOR_RGB2BGR))


def circular_pad_rgb(image_rgb: np.ndarray, pad_ratio: float) -> tuple[np.ndarray, int]:
    ensure_np()
    pad_width = int(image_rgb.shape[1] * pad_ratio)
    if pad_width <= 0:
        return image_rgb, 0
    return np.concatenate([image_rgb[:, -pad_width:], image_rgb, image_rgb[:, :pad_width]], axis=1), pad_width


def overlay_control_on_panorama(panorama_rgb: np.ndarray, control_rgb: np.ndarray, alpha: float) -> np.ndarray:
    ensure_np()
    mask = np.sum(control_rgb, axis=-1) != 0
    overlay = panorama_rgb.copy()
    overlay[mask] = np.clip(
        overlay[mask].astype(np.float32) * (1.0 - alpha) + control_rgb[mask].astype(np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)
    return overlay


def candidate_asset_names(kind: str, panorama_path: Path) -> list[str]:
    stem = panorama_path.stem
    suffixes_by_kind = {
        "segmentation": ["seg_stanford13_l3", "seg_sam_l3"],
        "surface_normal": ["normal_moge"],
        "wall_mask": ["wall_mask2former"],
    }
    suffixes = suffixes_by_kind[kind]
    names: list[str] = []
    for base in [stem, "panoImage_2048", "pano", "place_image"]:
        for suffix in suffixes:
            names.append(f"{base}_{suffix}.png")
    if kind == "segmentation":
        names.extend(["pano_seg_stanford13_l3.png", "pano_seg_sam_l3.png"])
    return list(dict.fromkeys(names))


def find_asset_path(explicit: str | None, panorama_path: Path, kind: str, required: bool = True) -> Path | None:
    if explicit:
        path = Path(explicit)
        if path.is_dir():
            for name in candidate_asset_names(kind, panorama_path):
                candidate = path / name
                if candidate.is_file():
                    return candidate
        if path.is_file():
            return path
        if required:
            raise FileNotFoundError(str(path))
        return None

    for name in candidate_asset_names(kind, panorama_path):
        candidate = panorama_path.parent / name
        if candidate.is_file():
            return candidate
    if required:
        raise FileNotFoundError(f"missing {kind} asset next to {panorama_path}")
    return None


STANFORD_TO_CORE = {
    "window": 4002,
    "bookcase": 5008,
    "sofa": 5002,
    "table": 5009,
    "chair": 5012,
    "door": 4001,
}
DEFAULT_CONTROL_MODEL_ROOT = Path(__file__).resolve().parent / "models" / "control_models"


@dataclass(frozen=True)
class ControlModelPaths:
    """Local model/code paths used to generate control assets."""

    model_root: Path = DEFAULT_CONTROL_MODEL_ROOT
    panosamic_root: Path | None = None
    panosamic_checkpoint: Path | None = None
    panosamic_config: Path | None = None
    sam_weights: Path | None = None
    moge_root: Path | None = None
    moge_model: Path | None = None
    mmdet_root: Path | None = None
    mask2former_weights: Path | None = None
    mask2former_model: str = "mask2former_swin-l-p4-w12-384-in21k_16xb1-lsj-100e_coco-panoptic"

    def resolved(self) -> "ControlModelPaths":
        root = Path(self.model_root).expanduser().resolve()
        return ControlModelPaths(
            model_root=root,
            panosamic_root=self._path(self.panosamic_root, root / "panosamic" / "PanoSAMic"),
            panosamic_checkpoint=self._path(
                self.panosamic_checkpoint,
                root / "panosamic" / "stanford2d3ds-vith-rgb-fold1",
            ),
            panosamic_config=self._path(
                self.panosamic_config,
                root / "panosamic" / "config_stanford2d3ds_dv.json",
            ),
            sam_weights=self._path(self.sam_weights, root / "panosamic" / "sam_vit_h_4b8939.pth"),
            moge_root=self._path(self.moge_root, root / "moge" / "MoGe"),
            moge_model=self._path(self.moge_model, root / "moge" / "moge-2-vitl-normal" / "model.pt"),
            mmdet_root=self._path(self.mmdet_root, root / "mmdetection"),
            mask2former_weights=self._path(
                self.mask2former_weights,
                root / "mmdetection" / "mask2former_swin-l-p4-w12-384-in21k_16xb1-lsj-100e_coco-panoptic.pth",
            ),
            mask2former_model=self.mask2former_model,
        )

    @staticmethod
    def _path(value: Path | None, default: Path) -> Path:
        return Path(value).expanduser().resolve() if value else default


@dataclass(frozen=True)
class ControlGenerationOptions:
    output_size: tuple[int, int] = (2048, 1024)
    infer_size: tuple[int, int] = (1024, 512)
    normal_batch_size: int = 12
    wall_batch_size: int = 4
    wall_id: int = 131
    device: str = "cuda:0"
    control_seed: int = 0
    canny_regions: int = 10
    canny_min_side_ratio: float = 0.1
    base_params_config: str | Path | None = None
    overwrite_assets: bool = False
    overwrite_geometric_proxy: bool = False


def require_path(path: Path | None, label: str) -> Path:
    if path is None or not Path(path).exists():
        raise FileNotFoundError(f"{label} is missing: {path}")
    return Path(path)


def encode_tagsys_instance(tagsys_id: int, instance_id: int) -> tuple[int, int, int]:
    padded = str(int(tagsys_id)).ljust(6, "0")
    primary_id = int(padded[0])
    secondary_id = int(padded[2:4])
    third_id = int(padded[4:6])
    bits = (
        format(primary_id, "03b")
        + format(secondary_id, "08b")
        + format(third_id, "06b")
        + format(int(instance_id) % 128, "07b")
    )
    value = int(bits, 2)
    return (value >> 16) & 255, (value >> 8) & 255, value & 255


def image_to_tensor(image: Image.Image, device: str):
    import numpy as np
    import torch

    array = np.ascontiguousarray(np.array(image).transpose(2, 0, 1), dtype=np.float32)
    return torch.as_tensor(array, device=device)


def connected_components(mask) -> list[Any]:
    import cv2

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype("uint8"), 8)
    components = []
    for idx in range(1, count):
        components.append((int(stats[idx, cv2.CC_STAT_AREA]), labels == idx))
    components.sort(key=lambda item: item[0], reverse=True)
    return [component for _, component in components]


def build_pseudo_segmentation(labels, class_names: list[str], min_component_area: int = 20, max_instances: int = 127):
    import numpy as np

    encoded = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for class_name, tagsys_id in STANFORD_TO_CORE.items():
        if class_name not in class_names:
            continue
        class_idx = class_names.index(class_name)
        kept = 0
        for component in connected_components(labels == class_idx):
            if int(component.sum()) < min_component_area:
                continue
            kept += 1
            if kept > max_instances:
                break
            encoded[component] = encode_tagsys_instance(tagsys_id, kept)
    return encoded


class SingleImageControlGenerator:
    """Generate seg/normal/wall assets and synthesize one geometric proxy."""

    def __init__(self, paths: ControlModelPaths, options: ControlGenerationOptions) -> None:
        self.paths = paths.resolved()
        self.options = options
        self._seg_model = None
        self._seg_class_names: list[str] | None = None
        self._normal_context = None
        self._wall_inferencer = None

    def close(self) -> None:
        self._seg_model = None
        self._normal_context = None
        self._wall_inferencer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def generate(
        self,
        white_model_panorama: str | Path,
        sample_dir: str | Path,
        geometric_proxy_path: str | Path | None = None,
    ) -> dict[str, str]:
        white_model_path = Path(white_model_panorama).expanduser().resolve()
        if not white_model_path.is_file():
            raise FileNotFoundError(str(white_model_path))
        out_dir = Path(sample_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        proxy_path = Path(geometric_proxy_path).expanduser().resolve() if geometric_proxy_path else out_dir / "geometric_proxy.png"
        assets = ControlAssetPaths(
            panorama=white_model_path,
            segmentation=out_dir / "segmentation.png",
            surface_normal=out_dir / "surface_normal.png",
            wall_mask=out_dir / "wall_mask.png",
        )

        if self.options.overwrite_assets or not assets.segmentation.is_file():
            self.generate_segmentation(white_model_path, assets.segmentation)
        if self.options.overwrite_assets or not assets.surface_normal.is_file():
            self.generate_surface_normal(white_model_path, assets.surface_normal)
        if self.options.overwrite_assets or not assets.wall_mask.is_file():
            self.generate_wall_mask(white_model_path, assets.wall_mask)
        if self.options.overwrite_geometric_proxy or not proxy_path.is_file():
            self.synthesize(assets, proxy_path)

        return {
            "white_model_panorama": str(white_model_path),
            "segmentation": str(assets.segmentation),
            "surface_normal": str(assets.surface_normal),
            "wall_mask": str(assets.wall_mask),
            "geometric_proxy": str(proxy_path),
        }

    def load_segmentation_model(self):
        if self._seg_model is not None:
            return self._seg_model, self._seg_class_names
        paths = self.paths
        sys.path.insert(0, str(require_path(paths.panosamic_root, "panosamic_root")))
        import enum
        import torch

        if not hasattr(enum, "member"):
            enum.member = lambda value: value
        from panosamic.datasets.stanford2d3ds import Stanford2d3dsDataset
        from panosamic.evaluation.utils.config import parse_modalities
        from panosamic.model import PanoSAMic

        model = PanoSAMic.from_pretrained_panosamic(
            str(require_path(paths.panosamic_checkpoint, "panosamic_checkpoint")),
            sam_weights_path=str(require_path(paths.sam_weights, "sam_weights")),
            vit_model="vit_h",
            config_path=str(require_path(paths.panosamic_config, "panosamic_config")),
            num_classes=len(Stanford2d3dsDataset.CLASS_NAMES),
            modalities=parse_modalities("image"),
        )
        self._seg_model = model.to(torch.device(self.options.device)).eval()
        self._seg_class_names = list(Stanford2d3dsDataset.CLASS_NAMES)
        return self._seg_model, self._seg_class_names

    def generate_segmentation(self, image_path: Path, output_path: Path) -> None:
        import numpy as np
        import torch

        model, class_names = self.load_segmentation_model()
        from panosamic.model.instance_semantic_fusion import refine_semantic_with_instances

        image = Image.open(image_path).convert("RGB")
        infer_image = image.resize(self.options.infer_size, Image.Resampling.BILINEAR)
        tensor = image_to_tensor(infer_image, self.options.device)
        with torch.no_grad():
            output = model([{"image": tensor}])[0]
        semantic_prediction = output["sem_preds"]
        instances = output.get("instance_masks") or []
        if instances:
            semantic_prediction = refine_semantic_with_instances(semantic_prediction.squeeze(0), instances).unsqueeze(0)
        labels = torch.argmax(semantic_prediction, dim=1).squeeze(0).detach().cpu().numpy().astype(np.int64)
        if labels.shape[::-1] != image.size:
            labels = np.array(
                Image.fromarray(labels.astype(np.uint8)).resize(image.size, Image.Resampling.NEAREST),
                dtype=np.int64,
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(build_pseudo_segmentation(labels, class_names)).save(output_path)

    def load_normal_context(self):
        if self._normal_context is not None:
            return self._normal_context
        paths = self.paths
        sys.path.insert(0, str(require_path(paths.moge_root, "moge_root")))
        import torch
        import utils3d
        from moge.model.v2 import MoGeModel
        from moge.utils.panorama import get_panorama_cameras, merge_panorama_normal, split_panorama_image
        from moge.utils.vis import colorize_normal

        model = MoGeModel.from_pretrained(str(require_path(paths.moge_model, "moge_model"))).to(torch.device(self.options.device)).eval()
        extrinsics, intrinsics = get_panorama_cameras()
        self._normal_context = (model, extrinsics, intrinsics, torch, utils3d, split_panorama_image, merge_panorama_normal, colorize_normal)
        return self._normal_context

    def generate_surface_normal(self, image_path: Path, output_path: Path) -> None:
        import cv2
        import numpy as np

        model, extrinsics, intrinsics, torch, utils3d, split_panorama_image, merge_panorama_normal, colorize_normal = self.load_normal_context()
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(str(image_path))
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.options.infer_size, cv2.INTER_AREA)
        height, width = image.shape[:2]
        split_images = split_panorama_image(image, extrinsics, intrinsics, 512)
        normal_maps = []
        masks = []
        with torch.inference_mode():
            for start in range(0, len(split_images), self.options.normal_batch_size):
                batch = split_images[start : start + self.options.normal_batch_size]
                tensor = torch.tensor(np.stack(batch) / 255, dtype=torch.float32, device=self.options.device).permute(0, 3, 1, 2)
                fov_x, _ = np.rad2deg(utils3d.numpy.intrinsics_to_fov(np.array(intrinsics[start : start + len(batch)])))
                fov_x = torch.tensor(fov_x, dtype=torch.float32, device=self.options.device)
                output = model.infer(tensor, fov_x=fov_x, apply_mask=False)
                masks.extend(list(output["mask"].detach().cpu().numpy()))
                normal = output.get("normal")
                if normal is None:
                    normal_maps.extend([np.zeros((512, 512, 3), dtype=np.float32) for _ in batch])
                else:
                    normal_maps.extend(list(normal.detach().cpu().numpy()))
        pano_normal, pano_mask = merge_panorama_normal(width, height, normal_maps, masks, extrinsics, intrinsics)
        norms = np.linalg.norm(pano_normal, axis=-1, keepdims=True)
        pano_normal = np.where(norms > 1e-6, pano_normal / norms, 0)
        out_rgb = colorize_normal(pano_normal, pano_mask)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(output_path), cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError(f"cv2.imwrite failed: {output_path}")

    def load_wall_inferencer(self):
        if self._wall_inferencer is not None:
            return self._wall_inferencer
        paths = self.paths
        mmdet_root = require_path(paths.mmdet_root, "mmdet_root")
        panopticapi_root = mmdet_root / "panopticapi"
        if panopticapi_root.exists():
            sys.path.insert(0, str(panopticapi_root))
        sys.path.insert(0, str(mmdet_root))
        from mmdet.apis import DetInferencer

        self._wall_inferencer = DetInferencer(
            model=paths.mask2former_model,
            weights=str(require_path(paths.mask2former_weights, "mask2former_weights")),
            device=self.options.device,
            show_progress=False,
        )
        if hasattr(self._wall_inferencer, "model"):
            self._wall_inferencer.model.float().eval()
        return self._wall_inferencer

    def generate_wall_mask(self, image_path: Path, output_path: Path) -> None:
        import cv2
        import numpy as np
        import torch

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(str(image_path))
        image = cv2.resize(image, self.options.infer_size)
        inferencer = self.load_wall_inferencer()
        device_type = "cuda" if str(self.options.device).startswith("cuda") and torch.cuda.is_available() else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            output = inferencer([image], batch_size=1, show=False)
        prediction = output["predictions"][0]
        wall_mask = (prediction["panoptic_seg"][:, :, 0] == self.options.wall_id).astype(np.uint8) * 255
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(output_path), wall_mask)
        if not ok:
            raise RuntimeError(f"cv2.imwrite failed: {output_path}")

    def synthesize(self, assets: ControlAssetPaths, output_path: Path) -> None:
        import numpy as np

        np.random.seed(self.options.control_seed)
        control_rgb, _stats = synthesize_geometry_control(
            assets,
            load_base_params_config(self.options.base_params_config),
            ControlSynthesisOptions(
                output_size=self.options.output_size,
                canny_regions=self.options.canny_regions,
                canny_min_side_ratio=self.options.canny_min_side_ratio,
            ),
        )
        write_rgb_image(output_path, control_rgb)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate segmentation, normal, wall mask, and geometric proxy from one white-model panorama."
    )
    parser.add_argument("--white-model-panorama", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", default="geometric_proxy.png")
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--control-width", type=int, default=2048)
    parser.add_argument("--control-height", type=int, default=1024)
    parser.add_argument("--normal-batch-size", type=int, default=12)
    parser.add_argument("--wall-batch-size", type=int, default=4)
    parser.add_argument("--wall-id", type=int, default=131)
    parser.add_argument("--base-params-config", default=DEFAULT_BASE_PARAMS_CONFIG)
    parser.add_argument("--control-seed", type=int, default=0)
    parser.add_argument("--canny-regions", type=int, default=10)
    parser.add_argument("--canny-min-side-ratio", type=float, default=0.1)
    parser.add_argument("--overwrite-assets", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def build_model_paths(args: argparse.Namespace) -> ControlModelPaths:
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


def build_generation_options(args: argparse.Namespace) -> ControlGenerationOptions:
    return ControlGenerationOptions(
        output_size=(args.control_width, args.control_height),
        normal_batch_size=args.normal_batch_size,
        wall_batch_size=args.wall_batch_size,
        wall_id=args.wall_id,
        device=args.device,
        control_seed=args.control_seed,
        canny_regions=args.canny_regions,
        canny_min_side_ratio=args.canny_min_side_ratio,
        base_params_config=args.base_params_config,
        overwrite_assets=args.overwrite_assets,
        overwrite_geometric_proxy=args.overwrite,
    )


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    proxy_path = output_dir / args.output_name
    generator = SingleImageControlGenerator(build_model_paths(args), build_generation_options(args))
    try:
        result = generator.generate(args.white_model_panorama, output_dir, proxy_path)
    finally:
        generator.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
