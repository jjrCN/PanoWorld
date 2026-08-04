"""PanoWorld panorama-normal fusion built on public MoGe view predictions.

Upstream MoGe provides panorama projection helpers and depth merging, but not
normal-map merging. Keeping this implementation here leaves the downloaded
MoGe checkout unmodified.
"""

from __future__ import annotations

import cv2
import numpy as np


def merge_panorama_normal(
    width: int,
    height: int,
    normal_maps: list[np.ndarray],
    pred_masks: list[np.ndarray],
    extrinsics: list[np.ndarray],
    intrinsics: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Merge perspective-view normal maps into one equirectangular normal map.

    Normals are transformed from camera to world coordinates and blended with
    view-angle and projection-center confidence weights.
    
    Args:
        width: Output panorama width
        height: Output panorama height  
        normal_maps: List of normal maps from different views (in camera space, range [-1,1])
        pred_masks: List of prediction masks for each view
        extrinsics: List of camera extrinsics for each view
        intrinsics: List of camera intrinsics for each view
        
    Returns:
        panorama_normal: Merged panorama normal map in world space
        panorama_mask: Valid region mask
    """
    if max(width, height) > 512:  # Increased threshold for better quality
        panorama_normal_init, _ = merge_panorama_normal(width // 2, height // 2, normal_maps, pred_masks, extrinsics, intrinsics)
        panorama_normal_init = cv2.resize(panorama_normal_init, (width, height), cv2.INTER_LINEAR)
        # Renormalize after resize with robust handling
        norms = np.linalg.norm(panorama_normal_init, axis=-1, keepdims=True)
        norms = np.where(np.isfinite(norms), norms, 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized_init = panorama_normal_init / norms
            normalized_init = np.where(np.isfinite(normalized_init), normalized_init, 0)
        panorama_normal_init = np.where(norms > 1e-6, normalized_init, 0)
    else:
        panorama_normal_init = None

    import utils3d
    from moge.utils.panorama import spherical_uv_to_directions

    uv = utils3d.numpy.image_uv(width=width, height=height)
    spherical_directions = spherical_uv_to_directions(uv)

    # Accumulate world-space normal vectors with confidence weighting
    panorama_normal_sum = np.zeros((height, width, 3), dtype=np.float64)  # Use double precision
    panorama_confidence = np.zeros((height, width), dtype=np.float64)
    panorama_pred_masks = []
    
    for i in range(len(normal_maps)):
        projected_uv, projected_depth = utils3d.numpy.project_cv(spherical_directions, extrinsics=extrinsics[i], intrinsics=intrinsics[i])
        projection_valid_mask = (projected_depth > 0) & (projected_uv > 0).all(axis=-1) & (projected_uv < 1).all(axis=-1)
        
        projected_pixels = utils3d.numpy.uv_to_pixel(np.clip(projected_uv, 0, 1), width=normal_maps[i].shape[1], height=normal_maps[i].shape[0]).astype(np.float32)
        
        # Get valid mask for this view
        panorama_pred_mask = projection_valid_mask & (cv2.remap(pred_masks[i].astype(np.uint8), projected_pixels[..., 0], projected_pixels[..., 1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE) > 0)
        panorama_pred_masks.append(panorama_pred_mask)
        
        # Remap normal map to panorama coordinates
        panorama_normal_cam = np.zeros((height, width, 3), dtype=np.float32)
        for c in range(3):
            panorama_normal_cam[:, :, c] = np.where(
                projection_valid_mask, 
                cv2.remap(normal_maps[i][:, :, c], projected_pixels[..., 0], projected_pixels[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), 
                0
            )
        
        # Ensure normal vectors are normalized with robust handling of invalid values
        norms_cam = np.linalg.norm(panorama_normal_cam, axis=-1, keepdims=True)
        # Handle NaN and infinite values in norms
        norms_cam = np.where(np.isfinite(norms_cam), norms_cam, 0)
        # Avoid division by zero/small values and ensure no NaN propagation
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized_cam = panorama_normal_cam / norms_cam
            # Replace any NaN or infinite results with zero
            normalized_cam = np.where(np.isfinite(normalized_cam), normalized_cam, 0)
        panorama_normal_cam = np.where(norms_cam > 1e-6, normalized_cam, 0)
        
        # Transform normals from camera space to world space
        # Use inverse transpose for normal transformation (R^T for orthogonal matrices)
        R = extrinsics[i][:3, :3]  # Camera rotation matrix (world to camera)
        R_inv = R.T  # Inverse rotation (camera to world)
        panorama_normal_world = np.einsum('ij,hwj->hwi', R_inv, panorama_normal_cam)
        
        # Compute confidence weights based on viewing angle and distance from projection center
        # Robust normalization of view directions
        view_norms = np.linalg.norm(spherical_directions, axis=-1, keepdims=True)
        view_norms = np.where(np.isfinite(view_norms) & (view_norms > 1e-12), view_norms, 1.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            view_directions = spherical_directions / view_norms
            view_directions = np.where(np.isfinite(view_directions), view_directions, 0)
        camera_forward = R_inv[:, 2]  # Camera forward direction in world space
        viewing_angle_weight = np.maximum(0, np.sum(view_directions * camera_forward, axis=-1))
        
        # Distance-based weight (closer to image center gets higher weight)
        center_distance = np.linalg.norm(projected_uv - 0.5, axis=-1)
        distance_weight = np.exp(-center_distance * 2)  # Gaussian falloff
        
        # Combined confidence
        confidence = panorama_pred_mask.astype(np.float64) * viewing_angle_weight * distance_weight
        
        # Accumulate normals with confidence weighting
        valid_mask = panorama_pred_mask & (norms_cam.squeeze() > 1e-6)
        panorama_normal_sum[valid_mask] += panorama_normal_world[valid_mask] * confidence[valid_mask, None]
        panorama_confidence[valid_mask] += confidence[valid_mask]
    
    # Normalize accumulated normals with robust division
    valid_pixels = panorama_confidence > 1e-6
    panorama_normal = np.zeros((height, width, 3), dtype=np.float32)
    if np.any(valid_pixels):
        # Robust division by confidence
        confidence_safe = panorama_confidence[valid_pixels, None]
        confidence_safe = np.where(confidence_safe > 1e-12, confidence_safe, 1e-12)
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized_result = panorama_normal_sum[valid_pixels] / confidence_safe
            normalized_result = np.where(np.isfinite(normalized_result), normalized_result, 0)
        panorama_normal[valid_pixels] = normalized_result.astype(np.float32)
    
    # Final normalization to unit vectors with robust handling
    norms = np.linalg.norm(panorama_normal, axis=-1, keepdims=True)
    norms = np.where(np.isfinite(norms), norms, 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized_final = panorama_normal / norms
        normalized_final = np.where(np.isfinite(normalized_final), normalized_final, 0)
    panorama_normal = np.where(norms > 1e-6, normalized_final, 0)
    
    # Apply smooth blending at boundaries using guided filtering
    if np.any(valid_pixels):
        panorama_normal = _smooth_normal_boundaries(panorama_normal, panorama_confidence, width, height)
    
    panorama_mask = np.any(panorama_pred_masks, axis=0)
    
    return panorama_normal, panorama_mask


def _smooth_normal_boundaries(normal_map: np.ndarray, confidence_map: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Apply smooth boundary blending for normal maps using guided filtering approach.
    This reduces seam artifacts while preserving normal vector properties.
    """
    # Convert confidence to a smooth guide map
    guide = cv2.GaussianBlur(confidence_map.astype(np.float32), (0, 0), max(width, height) / 100.0)
    guide = np.clip(guide, 0, 1)
    
    # Apply guided filter to each channel separately
    smoothed_normal = normal_map.copy()
    
    for c in range(3):
        channel = normal_map[:, :, c]
        
        # Only smooth where we have low confidence or boundary regions
        # Detect boundary regions where confidence changes rapidly
        grad_x = np.abs(np.gradient(confidence_map, axis=1))
        grad_y = np.abs(np.gradient(confidence_map, axis=0))
        boundary_mask = (grad_x + grad_y) > (np.max(grad_x + grad_y) * 0.1)
        
        # Apply mild Gaussian smoothing only in boundary regions
        if np.any(boundary_mask):
            smoothed_channel = cv2.GaussianBlur(channel, (0, 0), 1.0)
            # Blend original and smoothed based on boundary mask
            blend_weight = boundary_mask.astype(np.float32) * 0.3  # Mild blending
            smoothed_normal[:, :, c] = channel * (1 - blend_weight) + smoothed_channel * blend_weight
    
    # Handle panorama horizontal boundary (left-right wrap)
    # Smooth the transition at x=0 and x=width-1
    if width > 1:
        # Create smooth transition at horizontal boundaries
        left_border = smoothed_normal[:, :3, :]  # First 3 columns
        right_border = smoothed_normal[:, -3:, :]  # Last 3 columns
        
        # Average the borders for seamless wrapping
        avg_border = (left_border + right_border) / 2
        
        # Apply smooth transition
        for i in range(3):
            weight = (2 - i) / 3.0  # Decreasing weight: 1.0, 0.67, 0.33
            smoothed_normal[:, i, :] = smoothed_normal[:, i, :] * (1 - weight) + avg_border[:, i, :] * weight
            smoothed_normal[:, -(i+1), :] = smoothed_normal[:, -(i+1), :] * (1 - weight) + avg_border[:, -(i+1), :] * weight
    
    # Final normalization to ensure unit vectors with robust handling
    norms = np.linalg.norm(smoothed_normal, axis=-1, keepdims=True)
    norms = np.where(np.isfinite(norms), norms, 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized_smooth = smoothed_normal / norms
        normalized_smooth = np.where(np.isfinite(normalized_smooth), normalized_smooth, 0)
    smoothed_normal = np.where(norms > 1e-6, normalized_smooth, 0)
    
    return smoothed_normal
         
