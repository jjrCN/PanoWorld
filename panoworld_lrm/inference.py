import importlib
import os
import warnings


import numpy as np
import py360convert
import torch
from torch.utils.data import DataLoader
from PIL import Image
from skimage.metrics import structural_similarity

from .metric_utils import export_results
from .setup import init_config
from .utils import export_ply_forviewer, prepare_viewer

AMP_DTYPE_MAPPING = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "tf32": torch.float32,
}


def load_symbol(dotted_path):
    module_name, symbol_name = dotted_path.rsplit(".", 1)
    return importlib.import_module(module_name).__dict__[symbol_name]


def build_dataloader(config):
    dataset_cls = load_symbol(config.inference.get("dataset_name", "panoworld_lrm.dataset.Dataset"))
    dataset = dataset_cls(config)

    panoworld_mode = config.data.get("panoworld_mode", False)
    num_workers = 0 if panoworld_mode else config.inference.num_workers
    batch_size = 1 if panoworld_mode else config.inference.batch_size_per_gpu
    dataloader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
        "pin_memory": False,
    }
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = config.inference.prefetch_factor

    return DataLoader(dataset, **dataloader_kwargs)


def build_model(config, device):
    model_cls = load_symbol(config.model.class_name)
    model = model_cls(config).to(device)
    msg = model.load_ckpt(config.inference.ckpt_path)
    print(msg)
    model.eval()
    return model


def export_viewer_assets(result, out_dir, sh_degree):
    batch_size = result.input["input_images"].size(0)
    for batch_idx in range(batch_size):
        scene_name = result.input["input_target_scene_name"][batch_idx]
        inputs_view_name = result.input["input_view_names"][batch_idx]
        viewerdir = os.path.join(out_dir, f"{scene_name}/{inputs_view_name}/output_ply")
        point_cloud_dir = os.path.join(viewerdir, "point_cloud/iteration_0")
        os.makedirs(point_cloud_dir, exist_ok=True)

        export_ply_forviewer(
            result.gs_params,
            result.input["input_masks"][batch_idx],
            batch_idx,
            os.path.join(point_cloud_dir, "point_cloud.ply"),
        )
        prepare_viewer(result, viewerdir, sh_degree)


def batch_value(value):
    if isinstance(value, str):
        return value
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value[0].item()
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return ""
        return batch_value(value[0])
    return value


def batch_bool(value):
    value = batch_value(value)
    if isinstance(value, str):
        return value.lower() in ("1", "true", "yes")
    return bool(value)


def split_batch_string(value):
    value = batch_value(value)
    if value is None or value == "":
        return []
    return str(value).split("||")


def batch_item(value, batch_idx):
    if isinstance(value, str):
        if batch_idx == 0:
            return value
        raise IndexError(f"String batch field has no item {batch_idx}.")
    if isinstance(value, torch.Tensor):
        item = value.detach().cpu()[batch_idx]
        if item.numel() == 1:
            return item.item()
        return item
    if isinstance(value, (list, tuple)):
        return value[batch_idx]
    return value


def split_batch_item(value, batch_idx):
    value = batch_item(value, batch_idx)
    if value is None or value == "":
        return []
    return str(value).split("||")


def pano_stem(config):
    return config.data.pano_image_name.split(".")[0]


def render_result_panos(result):
    rendered_image = result.render
    rendered_depth = result.depth
    if rendered_image.size(0) != 1:
        raise ValueError("Panoworld post-processing expects batch_size=1.")

    _, num_faces, _, h, _ = rendered_image.size()
    if num_faces % 6 != 0:
        raise ValueError(f"Rendered target face count must be a multiple of 6, got {num_faces}.")

    pano_count = num_faces // 6
    pano_images = []
    pano_depths = []
    image_faces_np = []
    depth_faces_np = []
    for face_idx in range(num_faces):
        img_tensor = rendered_image[0, face_idx].detach().cpu().clamp(0, 1)
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        image_faces_np.append(img_np)

        depth_tensor = rendered_depth[0, face_idx].detach().cpu().float()
        depth_np = depth_tensor.permute(1, 2, 0).numpy()
        depth_faces_np.append(depth_np)

    for pano_idx in range(pano_count):
        image_faces = [image_faces_np[pano_idx * 6 + face_idx] for face_idx in range(6)]
        depth_faces = [depth_faces_np[pano_idx * 6 + face_idx] for face_idx in range(6)]
        pano_images.append(py360convert.c2e(image_faces, h=h, w=2 * h, cube_format="list"))
        pano_depth = py360convert.c2e(depth_faces, h=h, w=2 * h, cube_format="list")
        if pano_depth.ndim == 3:
            pano_depth = pano_depth[:, :, 0]
        pano_depths.append(pano_depth.astype(np.float32))

    return pano_images, pano_depths


def load_metric_image(path, target_size=None):
    image = Image.open(path).convert("RGB")
    if target_size is not None and image.size != target_size:
        image = image.resize(target_size, resample=Image.BICUBIC)
    return np.asarray(image).astype(np.float32) / 255.0


def psnr_from_mse(mse):
    if mse <= 1e-12:
        return float("inf")
    return float(-10.0 * np.log10(mse))


class LrmMetricEvaluator:
    def __init__(self, device, lpips_network="alex"):
        self.device = device
        self.lpips_network = lpips_network
        self.lpips_model = None
        self.values = []

    def _build_lpips(self):
        if self.lpips_model is None:
            import lpips

            self.lpips_model = lpips.LPIPS(net=self.lpips_network).to(self.device)
            self.lpips_model.eval()

    def add_pair(self, gt_path, render_path):
        if not os.path.exists(gt_path):
            print(f"Warning: missing GT image for metrics: {gt_path}")
            return
        if not os.path.exists(render_path):
            print(f"Warning: missing rendered image for metrics: {render_path}")
            return

        render = load_metric_image(render_path)
        gt = load_metric_image(gt_path, target_size=(render.shape[1], render.shape[0]))

        mse = float(np.mean((render - gt) ** 2))
        psnr = psnr_from_mse(mse)
        ssim = float(
            structural_similarity(
                gt,
                render,
                channel_axis=-1,
                data_range=1.0,
            )
        )

        self._build_lpips()
        gt_tensor = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(self.device)
        render_tensor = torch.from_numpy(render).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            lpips_value = self.lpips_model(
                render_tensor * 2.0 - 1.0,
                gt_tensor * 2.0 - 1.0,
            )
        lpips_score = float(lpips_value.detach().cpu().item())
        self.values.append((psnr, ssim, lpips_score))

    def summarize(self):
        if len(self.values) == 0:
            return None
        values = np.asarray(self.values, dtype=np.float64)
        return {
            "num_pairs": int(values.shape[0]),
            "psnr": float(np.mean(values[:, 0])),
            "ssim": float(np.mean(values[:, 1])),
            "lpips": float(np.mean(values[:, 2])),
        }


def build_lrm_metric_pairs(result, exported_panos):
    pairs = []
    input_data = result.input
    gt_paths_field = input_data.get("input_gt_pano_paths", None)
    if gt_paths_field is None:
        return pairs

    for exported in exported_panos:
        batch_idx = exported["batch_idx"]
        source_view_names = str(batch_item(input_data["input_source_view_names"], batch_idx)).split("-")
        gt_paths = split_batch_item(gt_paths_field, batch_idx)
        gt_by_view = dict(zip(source_view_names, gt_paths))
        gt_path = gt_by_view.get(exported["view_name"])
        if gt_path is None:
            print(f"Warning: no GT path found for view {exported['view_name']}; skipping metrics.")
            continue
        pairs.append((gt_path, exported["render_path"]))

    return pairs


def print_lrm_metrics(summary):
    if summary is None:
        print("\nLRM evaluation metrics: no valid GT/render image pairs were found.\n")
        return

    print("\nLRM evaluation metrics over all rendered panoramas:")
    print(f"  Image pairs: {summary['num_pairs']}")
    print(f"  PSNR : {summary['psnr']:.4f}")
    print(f"  SSIM : {summary['ssim']:.4f}")
    print(f"  LPIPS: {summary['lpips']:.4f}\n")


def read_place_depth_meters(target_dir_name):
    place_depth_path = os.path.join(target_dir_name, "place_depth.png")
    place_depth_scale_path = os.path.join(target_dir_name, "place_depth_scale.txt")
    if not os.path.exists(place_depth_path):
        raise FileNotFoundError(place_depth_path)
    if not os.path.exists(place_depth_scale_path):
        raise FileNotFoundError(place_depth_scale_path)

    place_depth = np.array(Image.open(place_depth_path)).astype(np.float32)
    with open(place_depth_scale_path, "r", encoding="utf-8") as f:
        depth_scale = float(f.read().strip())
    if depth_scale <= 0:
        raise ValueError(f"Invalid place depth scale: {place_depth_scale_path} -> {depth_scale}")
    return place_depth / depth_scale, place_depth > 0


def save_masked_lrm_pano(result, target_dir_name, lrm_path, lrm_mask_path):
    pano_images, pano_depths = render_result_panos(result)
    if len(pano_images) != 1 or len(pano_depths) != 1:
        raise ValueError("Intermediate panoworld refinement expects exactly one rendered pano.")

    os.makedirs(os.path.dirname(lrm_path), exist_ok=True)
    lrm_image = pano_images[0].copy()
    predicted_depth = pano_depths[0]
    Image.fromarray(lrm_image).save(lrm_path)

    place_depth, place_valid_mask = read_place_depth_meters(target_dir_name)
    if place_depth.shape != predicted_depth.shape:
        predicted_depth = np.array(
            Image.fromarray(predicted_depth).resize(
                (place_depth.shape[1], place_depth.shape[0]),
                resample=Image.BILINEAR,
            )
        ).astype(np.float32)

    mask = place_valid_mask & np.isfinite(predicted_depth) & ((predicted_depth - place_depth) > 0.3)
    if mask.shape != lrm_image.shape[:2]:
        mask = np.array(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                (lrm_image.shape[1], lrm_image.shape[0]),
                resample=Image.NEAREST,
            )
        ) > 0

    lrm_mask_image = lrm_image.copy()
    lrm_mask_image[mask] = (255, 255, 255)
    Image.fromarray(lrm_mask_image).save(lrm_mask_path)
    print(f"Saved masked LRM pano: {lrm_mask_path}; masked pixels: {int(mask.sum())}")


def handle_panoworld_result(result, batch, config, panorama_generator):
    target_dirs = split_batch_string(batch["panoworld_target_dir_names"])
    is_final = batch_bool(batch["panoworld_is_final"])

    if is_final:
        target_view_names = split_batch_string(batch["panoworld_target_view_names"])
        print(f"Exporting final LRM assets for {len(target_view_names)} PanoWorld nodes.")
        return True

    if len(target_dirs) != 1:
        raise ValueError("PanoWorld refinement expects exactly one target directory per batch.")

    target_dir_name = batch_value(batch["panoworld_target_dir_name"]) or target_dirs[0]
    has_deal_nearest_dir_name = batch_value(batch["panoworld_ref_view_dir_name"])
    ref_pano = batch_value(batch["panoworld_ref_pano_path"])

    stem = pano_stem(config)
    placeimg = os.path.join(target_dir_name, "place_image.png")
    lrm_path = os.path.join(target_dir_name, stem + "_lrm.png")
    ref_3d = os.path.join(target_dir_name, stem + "_lrm_mask.png")
    save_path = os.path.join(target_dir_name, config.data.pano_image_name)

    save_masked_lrm_pano(result, target_dir_name, lrm_path, ref_3d)

    for required_path in (placeimg, ref_3d, ref_pano):
        if not os.path.exists(required_path):
            raise FileNotFoundError(required_path)

    print(f"Refining {target_dir_name}; nearest fixed view dir: {has_deal_nearest_dir_name}; ref pano: {ref_pano}")
    panorama_generator.generate(
        white_model_panorama=placeimg,
        coarse_view=ref_3d,
        style_reference=ref_pano,
        output_path=save_path,
        prompt=config.data.get("panoworld_prompt", ""),
    )
    if not os.path.exists(save_path):
        raise FileNotFoundError(save_path)
    return False


def run_inference(config):
    os.environ["OMP_NUM_THREADS"] = str(config.inference.get("num_threads", 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cuda.matmul.allow_tf32 = config.inference.use_tf32
    torch.backends.cudnn.allow_tf32 = config.inference.use_tf32

    dataloader = build_dataloader(config)
    model = build_model(config, device)
    panorama_generator = None
    if config.data.get("panoworld_mode", False):
        from panoworld_pipeline.native_2d import NativePanoramaGenerator

        panorama_generator = NativePanoramaGenerator.from_config(config)

    print(f"Running inference; save results to: {config.inference.out_dir}")
    warnings.filterwarnings("ignore", category=FutureWarning)

    evaluation_folder_list = []
    metric_pairs = []
    autocast_enabled = config.inference.use_amp and device.type == "cuda"
    autocast_dtype = AMP_DTYPE_MAPPING[config.inference.amp_dtype]

    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        enabled=autocast_enabled,
        dtype=autocast_dtype,
    ):
        sample_target_images = config.data.get("sample_target_images", 6)
        for uid, batch in enumerate(dataloader, start=1):
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            print(uid - 1)

            input_data_dict = {key: value for key, value in batch.items() if "input" in key}
            target_data_dict = {key: value for key, value in batch.items() if "target" in key}
            result = model(input_data_dict, target_data_dict)

            should_export_assets = True
            if config.data.get("panoworld_mode", False):
                should_export_assets = handle_panoworld_result(result, batch, config, panorama_generator)

            if should_export_assets:
                exported_panos = export_results(
                    result,
                    config.inference.out_dir,
                    uid=uid,
                )
                if not config.data.get("panoworld_mode", False):
                    metric_pairs.extend(build_lrm_metric_pairs(result, exported_panos))
                for batch_idx in range(input_data_dict["input_images"].size(0)):
                    scene_name = result.input["input_target_scene_name"][batch_idx]
                    inputs_view_name = result.input["input_view_names"][batch_idx]
                    evaluation_folder_list.append(
                        os.path.join(config.inference.out_dir, f"{scene_name}/{inputs_view_name}")
                    )

                export_viewer_assets(result, config.inference.out_dir, config.model.gaussians.sh_degree)

    should_eval_metrics = (
        not config.data.get("panoworld_mode", False)
        and config.inference.get("eval_metrics", True)
    )
    if should_eval_metrics:
        del model
        if "result" in locals():
            del result
        if "batch" in locals():
            del batch

    if device.type == "cuda":
        torch.cuda.empty_cache()

    if should_eval_metrics:
        evaluator = LrmMetricEvaluator(
            device=device,
            lpips_network=config.inference.get("lpips_network", "alex"),
        )
        for gt_path, render_path in metric_pairs:
            evaluator.add_pair(gt_path, render_path)
        print_lrm_metrics(evaluator.summarize())


def main():
    run_inference(init_config())


if __name__ == "__main__":
    main()
