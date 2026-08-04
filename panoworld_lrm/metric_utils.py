import torch
from torch import Tensor
from jaxtyping import Float
from einops import reduce, rearrange
from skimage.metrics import structural_similarity
import functools
import os
from PIL import Image
import imageio
import numpy as np
from easydict import EasyDict as edict
import json
import torchvision
import py360convert
import warnings
import cv2
# Suppress warnings for LPIPS loss loading
warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'pretrained' is deprecated since 0.13")
warnings.filterwarnings("ignore", category=UserWarning, message="Arguments other than a weight enum.*")

def normalize_depth(depth_map):
        """
        Args:
            depth_map: (B, V, 1, H, W)
        Returns:
            normalized_depth: (B, V, 1, H, W) in range [0, 1]
        """
        B, V, C, H, W = depth_map.shape
        # Flatten H and W so min/max can be computed per image: (B, V, H*W)
        depth_flat = depth_map.view(B, V, -1)

        # Compute per-image maxima and minima
        max_val = depth_flat.max(dim=-1, keepdim=True)[0] # (B, V, 1)
        min_val = depth_flat.min(dim=-1, keepdim=True)[0] # (B, V, 1)

        # Restore dimensions for broadcasting: (B, V, 1, 1, 1)
        max_val = max_val.view(B, V, 1, 1, 1)
        min_val = min_val.view(B, V, 1, 1, 1)

        # Compute the denominator and guard against divide-by-zero
        denominator = max_val - min_val
        denominator = torch.where(denominator < 1e-6, torch.ones_like(denominator), denominator)

        return (depth_map - min_val) / denominator

@torch.no_grad()
def export_results(
    result: edict,
    out_dir: str,
    uid: int = 0
):
    """
    Save results including images and optional metrics and videos.

    Args:
        result: EasyDict containing input, target, and rendered images, and optionally video frames
        out_dir: Directory to save the evaluation results
    """
    os.makedirs(out_dir, exist_ok=True)

    target_data = result.target
    rendered_image = result.render
    rendered_depth = result.depth
    rendered_input_depth = result.depth_dist
    input_data = result.input
    b, v, _, h, w = rendered_image.size()
    t = input_data["input_depths"].size(1)

    exported_panos = []
    for batch_idx in range(input_data["input_images"].size(0)):
        scene_name = input_data["input_target_scene_name"][batch_idx]
        inputs_view_name = input_data["input_view_names"][batch_idx]

        sample_dir = os.path.join(out_dir, f"{scene_name}/{inputs_view_name}")
        inputs_view_name_list = inputs_view_name.split("-")

        os.makedirs(sample_dir, exist_ok=True)
        os.makedirs(os.path.join(sample_dir, "target_rendering"), exist_ok=True)
        os.makedirs(os.path.join(sample_dir, "target_rendering_depth"), exist_ok=True)

        faces_np = [] # Collect the 6 cube faces converted to NumPy arrays
        faces_depth_np = []
        for i in range(v):
            img_tensor = rendered_image[batch_idx, i].detach().cpu().clamp(0, 1)
            depth_tensor = rendered_depth[batch_idx, i].detach().cpu().clamp(0, 30)
            depth_np = depth_tensor.permute(1, 2, 0).numpy()
            img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            faces_np.append(img_np)
            faces_depth_np.append(depth_np)

        if v % 6 == 0:
            pano_image_num = v // 6
            for i_pano in range(pano_image_num):
                pano_height = h
                pano_width = 2*h
                faces_np_pano = [faces_np[i_pano*6+i] for i in range(6)]
                faces_np_pano_depth = [faces_depth_np[i_pano*6+i] for i in range(6)]
                panorama_np = py360convert.c2e(faces_np_pano, h=pano_height, w=pano_width, cube_format='list')
                panorama_depth_np = py360convert.c2e(faces_np_pano_depth, h=pano_height, w=pano_width, cube_format='list')
                depth_scale = int(65535.0 / panorama_depth_np.max() - 0.5)
                panorama_depth_np = (panorama_depth_np*depth_scale).astype(np.uint16)
                panorama_path = os.path.join(sample_dir, "target_rendering", inputs_view_name_list[i_pano] + ".png")
                panorama_depth_path = os.path.join(sample_dir, "target_rendering_depth", inputs_view_name_list[i_pano] + ".png")
                Image.fromarray(panorama_np).save(panorama_path)
                cv2.imwrite(panorama_depth_path, panorama_depth_np)
                exported_panos.append({
                    "batch_idx": batch_idx,
                    "view_name": inputs_view_name_list[i_pano],
                    "render_path": panorama_path,
                })
                with open(os.path.join(sample_dir, "rendering_depth_scale.txt"), "w", encoding="utf-8") as f:
                    f.write(str(depth_scale))

    return exported_panos
