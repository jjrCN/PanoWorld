import torch
import cv2
import os
import numpy as np
from collections import OrderedDict
from argparse import Namespace
import json
import math

from einops import rearrange
from plyfile import PlyData, PlyElement

def compute_plucmap_pano(c2w, h, w):
    """
    Generate rays for an equirectangular panorama (ERP).
    Args:
        c2w (torch.tensor): [b, v, 4, 4]
        h, w (int): Image height and width
    Returns:
        ray_o: (b, v, 3, h, w)
        ray_d: (b, v, 3, h, w)
    """
    b, v = c2w.size(0), c2w.size(1)
    device = c2w.device

    # 1. Generate the pixel grid [h, w]
    grid_x = torch.arange(w, device=device)
    grid_y = torch.arange(h, device=device)
    idx_y, idx_x = torch.meshgrid(grid_y, grid_x, indexing='ij') # [h, w]

    # 2. Map pixel coordinates to longitude and latitude
    # Longitude (phi): [0, w] -> [-pi, pi]
    # Latitude (theta): [0, h] -> [pi/2, -pi/2] (the top row usually corresponds to the north pole)
    lon = (idx_x + 0.5) / w * 2 * np.pi - np.pi
    lat = np.pi / 2 - (idx_y + 0.5) / h * np.pi

    # 3. Convert spherical coordinates to Cartesian coordinates in the local camera frame
    # Assume the default forward direction is +Z (lon=0, lat=0), with +X right and +Y up
    # Negate y to follow the computer-vision convention where image y points downward
    x = torch.cos(lat) * torch.sin(lon)
    y = -torch.sin(lat)
    z = torch.cos(lat) * torch.cos(lon)

    ray_d = torch.stack([x, y, z], dim=0) # [3, h, w]

    # 4. Expand to batched dimensions
    ray_d = ray_d.unsqueeze(0).unsqueeze(0).expand(b, v, -1, -1, -1) # [b, v, 3, h, w]
    ray_d = ray_d.reshape(b * v, 3, h * w)

    c2w = c2w.reshape(b * v, 4, 4)

    # 5. Rotate rays into the world coordinate frame
    ray_d = torch.bmm(c2w[:, :3, :3], ray_d) # [b*v, 3, h*w]
    # These panorama ray directions are already unit vectors in theory, but we renormalize for safety
    ray_d = ray_d / torch.norm(ray_d, dim=1, keepdim=True)

    # 6. Get ray origins
    ray_o = c2w[:, :3, 3:4].expand(b * v, -1, h * w) # [b*v, 3, h*w]

    # 7. Reshape back to the target layout
    ray_o = ray_o.reshape(b, v, 3, h, w)
    ray_d = ray_d.reshape(b, v, 3, h, w)

    return ray_o, ray_d

def compute_rays_pano(c2w, h, w):
    """
    Generate flattened rays for an equirectangular panorama.
    """
    ray_o, ray_d = compute_plucmap_pano(c2w, h, w)

    # Flatten to [b, v*h*w, 3]
    ray_o = rearrange(ray_o, 'b v c h w -> b (v h w) c')
    ray_d = rearrange(ray_d, 'b v c h w -> b (v h w) c')

    return ray_o, ray_d

def compute_plucmap(fxfycxcy, c2w, h, w):
    """Transform target before computing loss
    Args:
        fxfycxcy (torch.tensor): [b, v, 4]
        c2w (torch.tensor): [b, v, 4, 4]
    Returns:
        ray_o: (b, v, 3, h, w)
        ray_d: (b, v, 3, h, w)
    """
    b, v = fxfycxcy.size(0), fxfycxcy.size(1)

    # Efficient meshgrid equivalent using broadcasting
    idx_x = torch.arange(w, device=c2w.device)[None, :].expand(h, -1)  # [h, w]
    idx_y = torch.arange(h, device=c2w.device)[:, None].expand(-1, w)  # [h, w]

    # Reshape for batched matrix multiplication
    idx_x = idx_x.flatten().expand(b * v, -1)           # [b*v, h*w]
    idx_y = idx_y.flatten().expand(b * v, -1)           # [b*v, h*w]

    fxfycxcy = fxfycxcy.reshape(b * v, 4)               # [b*v, 4]
    c2w = c2w.reshape(b * v, 4, 4)                      # [b*v, 4, 4]

    x = (idx_x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]     # [b*v, h*w]
    y = (idx_y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]     # [b*v, h*w]
    z = torch.ones_like(x)                                      # [b*v, h*w]

    ray_d = torch.stack([x, y, z], dim=1)                       # [b*v, 3, h*w]
    ray_d = torch.bmm(c2w[:, :3, :3], ray_d)                    # [b*v, 3, h*w]
    ray_d = ray_d / torch.norm(ray_d, dim=1, keepdim=True)      # [b*v, 3, h*w]

    ray_o = c2w[:, :3, 3:4].expand(b * v, -1, h*w)              # [b*v, 3, h*w]

    ray_o = ray_o.reshape(b, v, 3, h, w)                        # [b, v, 3, h, w]
    ray_d = ray_d.reshape(b, v, 3, h, w)                        # [b, v, 3, h, w]

    return ray_o, ray_d

def compute_rays(fxfycxcy, c2w, h, w):
    """Transform target before computing loss
    Args:
        fxfycxcy (torch.tensor): [b, v, 4]
        c2w (torch.tensor): [b, v, 4, 4]
    Returns:
        ray_o: (b, v, 3, h, w)
        ray_d: (b, v, 3, h, w)
    """
    b, v = fxfycxcy.size(0), fxfycxcy.size(1)

    # Efficient meshgrid equivalent using broadcasting
    idx_x = torch.arange(w, device=c2w.device)[None, :].expand(h, -1)  # [h, w]
    idx_y = torch.arange(h, device=c2w.device)[:, None].expand(-1, w)  # [h, w]

    # Reshape for batched matrix multiplication
    idx_x = idx_x.flatten().expand(b * v, -1)           # [b*v, h*w]
    idx_y = idx_y.flatten().expand(b * v, -1)           # [b*v, h*w]

    fxfycxcy = fxfycxcy.reshape(b * v, 4)               # [b*v, 4]
    c2w = c2w.reshape(b * v, 4, 4)                      # [b*v, 4, 4]

    x = (idx_x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]     # [b*v, h*w]
    y = (idx_y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]     # [b*v, h*w]
    z = torch.ones_like(x)                                      # [b*v, h*w]

    ray_d = torch.stack([x, y, z], dim=1)                       # [b*v, 3, h*w]
    ray_d = torch.bmm(c2w[:, :3, :3], ray_d)                    # [b*v, 3, h*w]
    ray_d = ray_d / torch.norm(ray_d, dim=1, keepdim=True)      # [b*v, 3, h*w]

    ray_o = c2w[:, :3, 3:4].expand(b * v, -1, h*w)              # [b*v, 3, h*w]

    ray_o = ray_o.reshape(b, v, 3, h, w)                        # [b, v, 3, h, w]
    ray_d = ray_d.reshape(b, v, 3, h, w)                        # [b, v, 3, h, w]

    ray_o = rearrange(ray_o, 'b v c h w -> b (v h w) c')
    ray_d = rearrange(ray_d, 'b v c h w -> b (v h w) c')

    return ray_o, ray_d


def position_grid_to_embed(pos_grid: torch.Tensor, embed_dim: int, omega_0: float = 100) -> torch.Tensor:
    """
    Convert 2D position grid (HxWx2) to sinusoidal embeddings (HxWxC)

    Args:
        pos_grid: Tensor of shape (H, W, 2) containing 2D coordinates
        embed_dim: Output channel dimension for embeddings

    Returns:
        Tensor of shape (H, W, embed_dim) with positional embeddings
    """
    H, W, grid_dim = pos_grid.shape
    assert grid_dim == 2
    pos_flat = pos_grid.reshape(-1, grid_dim)  # Flatten to (H*W, 2)

    # Process x and y coordinates separately
    emb_x = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 0], omega_0=omega_0)  # [1, H*W, D/2]
    emb_y = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 1], omega_0=omega_0)  # [1, H*W, D/2]

    # Combine and reshape
    emb = torch.cat([emb_x, emb_y], dim=-1)  # [1, H*W, D]

    return emb.view(H, W, embed_dim)  # [H, W, D]


def make_sincos_pos_embed(embed_dim: int, pos: torch.Tensor, omega_0: float = 100) -> torch.Tensor:
    """
    This function generates a 1D positional embedding from a given grid using sine and cosine functions.

    Args:
    - embed_dim: The embedding dimension.
    - pos: The position to generate the embedding from.

    Returns:
    - emb: The generated 1D positional embedding.
    """
    assert embed_dim % 2 == 0
    device = pos.device
    omega = torch.arange(embed_dim // 2, dtype=torch.float32 if device.type == "mps" else torch.double, device=device)
    omega /= embed_dim / 2.0
    omega = 1.0 / omega_0**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb.float()


# Inspired by https://github.com/microsoft/moge


def create_uv_grid(
    width: int, height: int, aspect_ratio: float = None, dtype: torch.dtype = None, device: torch.device = None
) -> torch.Tensor:
    """
    Create a normalized UV grid of shape (width, height, 2).

    The grid spans horizontally and vertically according to an aspect ratio,
    ensuring the top-left corner is at (-x_span, -y_span) and the bottom-right
    corner is at (x_span, y_span), normalized by the diagonal of the plane.

    Args:
        width (int): Number of points horizontally.
        height (int): Number of points vertically.
        aspect_ratio (float, optional): Width-to-height ratio. Defaults to width/height.
        dtype (torch.dtype, optional): Data type of the resulting tensor.
        device (torch.device, optional): Device on which the tensor is created.

    Returns:
        torch.Tensor: A (width, height, 2) tensor of UV coordinates.
    """
    # Derive aspect ratio if not explicitly provided
    if aspect_ratio is None:
        aspect_ratio = float(width) / float(height)

    # Compute normalized spans for X and Y
    diag_factor = (aspect_ratio**2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag_factor
    span_y = 1.0 / diag_factor

    # Establish the linspace boundaries
    left_x = -span_x * (width - 1) / width
    right_x = span_x * (width - 1) / width
    top_y = -span_y * (height - 1) / height
    bottom_y = span_y * (height - 1) / height

    # Generate 1D coordinates
    x_coords = torch.linspace(left_x, right_x, steps=width, dtype=dtype, device=device)
    y_coords = torch.linspace(top_y, bottom_y, steps=height, dtype=dtype, device=device)

    # Create 2D meshgrid (width x height) and stack into UV
    uu, vv = torch.meshgrid(x_coords, y_coords, indexing="xy")
    uv_grid = torch.stack((uu, vv), dim=-1)

    return uv_grid

def export_ply_forviewer(gs_params, input_masks, batch_idx, filename):
    """
    Export Gaussians to the standard 3DGS PLY format that can be loaded by SIBR/Supersplat.
    """
    gs_mask = input_masks.detach().cpu().numpy().reshape(-1)

    # 1. Fetch data from the selected batch (typically batch size is 1 during export)
    # Convert tensors to CPU NumPy arrays
    xyz = gs_params["xyz"][batch_idx].detach().cpu().numpy()              # (N, 3)
    xyz = xyz[gs_mask]
    features = gs_params["feature"][batch_idx].detach().cpu().numpy()     # (N, (deg+1)^2, 3)
    features = features[gs_mask]
    scale = gs_params["scale"][batch_idx].detach().cpu().numpy()          # (N, 3)
    scale = scale[gs_mask]
    rotation = gs_params["rotation"][batch_idx].detach().cpu().numpy()    # (N, 4)
    rotation = rotation[gs_mask]
    # Normalize quaternion to avoid artifacts in some viewers
    norm = np.linalg.norm(rotation, axis=1, keepdims=True)
    rotation = rotation / (norm + 1e-6)
    opacity_raw = gs_params["opacity"][batch_idx].detach().cpu().numpy() # (T, N, 1)
    # opacity_raw = np.ones_like(gs_params["opacity"][0].detach().cpu().numpy()) # (T, N, 1)

    # 2. Handle opacity
    # The model predicts view-dependent opacity with shape [T, N, 1]
    # PLY is static, so we approximate it using the first view or an average
    # Using view 0 is usually sufficient for debugging and visualization
    if opacity_raw.ndim == 3:
        opacity = opacity_raw[0] # Use the first view: (N, 1)
    else:
        opacity = opacity_raw
    opacity = opacity.squeeze(-1) # (N,)
    opacity = opacity[gs_mask]
    # 3. Handle spherical harmonics (color)
    # features shape: (N, 16, 3) assuming sh_degree=3
    # The standard PLY layout splits coefficients into f_dc (order 0) and f_rest (orders 1-3)
    f_dc = features[:, 0, :] # (N, 3)

    # Handle higher-order coefficients (f_rest)
    # The standard 3DGS order moves color channels first, then flattens
    # Input: (N, 15, 3) -> transpose to (N, 3, 15) -> flatten to (N, 45)
    # Final order: [r1...r15, g1...g15, b1...b15]
    f_rest = features[:, 1:, :]
    f_rest = f_rest.transpose(0, 2, 1).reshape(f_rest.shape[0], -1)

    # 4. Build the PLY attribute layout
    dtype_full = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4')
    ]

    # Add attribute names for f_rest
    num_rest_coeffs = f_rest.shape[1] if f_rest.shape[1] > 0 else 0
    for i in range(num_rest_coeffs):
        dtype_full.append((f'f_rest_{i}', 'f4'))

    # Add opacity, scale, and rotation attributes
    dtype_full.append(('opacity', 'f4'))

    for i in range(3):
        dtype_full.append((f'scale_{i}', 'f4'))

    for i in range(4):
        dtype_full.append((f'rot_{i}', 'f4'))

    # 5. Fill the structured array
    elements = np.empty(xyz.shape[0], dtype=dtype_full)

    # Coordinates
    elements['x'] = xyz[:, 0]
    elements['y'] = xyz[:, 1]
    elements['z'] = xyz[:, 2]

    # Normals (typically set to zero)
    elements['nx'] = np.zeros_like(xyz[:, 0])
    elements['ny'] = np.zeros_like(xyz[:, 0])
    elements['nz'] = np.zeros_like(xyz[:, 0])

    # DC color coefficients
    elements['f_dc_0'] = f_dc[:, 0]
    elements['f_dc_1'] = f_dc[:, 1]
    elements['f_dc_2'] = f_dc[:, 2]

    # Higher-order color coefficients
    for i in range(num_rest_coeffs):
        elements[f'f_rest_{i}'] = f_rest[:, i]

    # The viewer applies sigmoid to opacity, so we keep the raw logits here.
    elements['opacity'] = opacity

    # Scale (the model stores scale in log-space, so save it directly)
    elements['scale_0'] = scale[:, 0]
    elements['scale_1'] = scale[:, 1]
    elements['scale_2'] = scale[:, 2]

    # Rotation (Quaternion: w, x, y, z)
    # The model outputs rotation as (N, 4); confirm the convention if needed.
    # Common gsplat/3DGS conventions use (w, x, y, z) or (r, x, y, z).
    # As long as the output follows a valid quaternion convention, it can be saved directly.
    elements['rot_0'] = rotation[:, 0]
    elements['rot_1'] = rotation[:, 1]
    elements['rot_2'] = rotation[:, 2]
    elements['rot_3'] = rotation[:, 3]

    # 6. Write the PLY file
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(filename)
    print(f"Saved PLY to {filename}")

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def prepare_viewer(result, dirname, sh_degree):
    #1. cfg_args
    input_dict = result.input
    target_dict = result.target
    cfg_dict = {}
    cfg_dict['source_path'] = '' # It does not matter
    cfg_dict['sh_degree'] = sh_degree
    cfg_dict['white_background'] = True
    with open(dirname+'/cfg_args', 'w') as f:
        f.write(str(Namespace(**cfg_dict)))
    #2. Camera pose
    cameras_towrite= []
    for b in range(input_dict.input_c2ws.shape[0]):
        for i in range(input_dict.input_c2ws.shape[1]):
            fx = target_dict.target_fxfycxcy[0][0][0].item()
            fy = target_dict.target_fxfycxcy[0][0][1].item()
            cx = target_dict.target_fxfycxcy[0][0][2].item()
            cy = target_dict.target_fxfycxcy[0][0][3].item()
            c2w  = input_dict.input_c2ws[b][i]
            width = int(fx) * 2
            height = int(fy) * 2
            cam = {'id':b*input_dict.input_c2ws.shape[1]+i, 'img_name':f'img_{b}_{i}.png',
                'width': width,
                'height': height,
                'fx': fx,
                'fy': fy,
                'FovX': None, 'FovY': None,
                'position': None, 'rotation': None}
            FovX = focal2fov(fx, width)
            FovY = focal2fov(fy, height)
            # c2w_colmap_4x4 = np.eye(4)
            c2w_colmap_4x4 = c2w.cpu().numpy()
            # c2w_colmap_4x4[:3,1:3]*=-1 #flip y and z
            w2c = np.linalg.inv(c2w_colmap_4x4)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            Rt = np.zeros((4, 4))
            Rt[:3, :3] = R.transpose()
            Rt[:3, 3] = T
            Rt[3, 3] = 1.0

            W2C = np.linalg.inv(Rt)
            pos = W2C[:3, 3]
            rot = W2C[:3, :3]
            serializable_array_2d = [x.tolist() for x in rot]
            cam['position'] = pos.tolist()
            cam['rotation'] = serializable_array_2d
            cameras_towrite.append(cam)
    with open(dirname+'/cameras.json', 'w') as f:
        json.dump(cameras_towrite, f)
