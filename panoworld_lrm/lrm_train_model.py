import torch
from torch import nn
from easydict import EasyDict as edict
from einops.layers.torch import Rearrange
from einops import rearrange
import traceback
from gsplat import rasterization
import torch.nn.functional as F
import os
from .transformer import TransformerBlock
from .utils import (
    compute_rays,
    compute_plucmap,
    compute_rays_pano,
    compute_plucmap_pano,
    export_ply_forviewer,
)
import numpy as np
from .dpt_head import DPTHead
from .torch_impl import _spherical_harmonics
from .prope_custom import PropeDotProductAttention
from .loss import LossComputer


def _init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.RMSNorm, nn.LayerNorm)):
        module.reset_parameters()
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class GaussianRenderer(torch.autograd.Function):
    @staticmethod
    def render(xyz, feature, scale, rotation, opacity, test_c2w, test_intr,
               W, H, sh_degree, near_plane, far_plane):
        opacity = opacity.sigmoid().squeeze(-1)
        scale = scale.exp()
        # rotation = F.normalize(rotation, p=2, dim=-1)
        # test_w2c = test_c2w.float().inverse().unsqueeze(0) # (1, 4, 4)
        try:
            test_w2c = test_c2w.float().inverse().unsqueeze(0)
        except RuntimeError as e:
            print(f"Error at Rank {torch.distributed.get_rank()}: Matrix is not invertible, test_c2w: {test_c2w}")
            exit(0)
        test_intr_i = torch.zeros(3, 3).to(test_intr.device)
        test_intr_i[0, 0] = test_intr[0]
        test_intr_i[1, 1] = test_intr[1]
        test_intr_i[0, 2] = test_intr[2]
        test_intr_i[1, 2] = test_intr[3]
        test_intr_i[2, 2] = 1
        test_intr_i = test_intr_i.unsqueeze(0) # (1, 3, 3)
        rendering, _, _ = rasterization(xyz, rotation, scale, opacity, feature,
                                        test_w2c, test_intr_i, W, H, sh_degree=sh_degree,
                                        near_plane=near_plane, far_plane=far_plane,
                                        packed=False,
                                        absgrad=False,
                                        sparse_grad=False,
                                        render_mode="RGB+ED",
                                        backgrounds=torch.ones(1, 3).to(test_intr.device),
                                        rasterize_mode='classic') # (1, H, W, 5)
        return rendering # (1, H, W, 4)

    @staticmethod
    def forward(ctx, xyz, feature, scale, rotation, opacity, test_c2ws, test_intr,
                W, H, sh_degree, near_plane, far_plane):
        ctx.save_for_backward(xyz, feature, scale, rotation, opacity, test_c2ws, test_intr)
        ctx.W = W
        ctx.H = H
        ctx.sh_degree = sh_degree
        ctx.near_plane = near_plane
        ctx.far_plane = far_plane
        with torch.no_grad():
            B, V, _ = test_intr.shape
            # Initialize 4-channel tensors for RGB(3) + depth(1).
            renderings = torch.zeros(B, V, H, W, 4).to(xyz.device)
            for ib in range(B):
                for iv in range(V):
                    renderings[ib, iv:iv+1] = GaussianRenderer.render(xyz[ib], feature[ib], scale[ib], rotation[ib], opacity[ib,iv],
                                                                      test_c2ws[ib,iv], test_intr[ib,iv], W, H, sh_degree, near_plane, far_plane)
        renderings = renderings.requires_grad_()
        return renderings

    @staticmethod
    def backward(ctx, grad_output):
        xyz, feature, scale, rotation, opacity, test_c2ws, test_intr = ctx.saved_tensors
        xyz = xyz.detach().requires_grad_()
        feature = feature.detach().requires_grad_()
        scale = scale.detach().requires_grad_()
        rotation = rotation.detach().requires_grad_()
        opacity = opacity.detach().requires_grad_()
        W = ctx.W
        H = ctx.H
        sh_degree = ctx.sh_degree
        near_plane = ctx.near_plane
        far_plane = ctx.far_plane
        with torch.enable_grad():
            B, V, _ = test_intr.shape
            for ib in range(B):
                for iv in range(V):
                    rendering = GaussianRenderer.render(xyz[ib], feature[ib], scale[ib], rotation[ib], opacity[ib,iv],
                                                        test_c2ws[ib,iv], test_intr[ib,iv], W, H, sh_degree, near_plane, far_plane)
                    rendering.backward(grad_output[ib, iv:iv+1])

        return xyz.grad, feature.grad, scale.grad, rotation.grad, opacity.grad, None, None, None, None, None, None, None


class PanoWorldLRMTraining(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim1 = config.model.dim1
        self.dim2 = config.model.dim2
        self.dim3 = config.model.dim3
        self.pose_keys = ["ray_o", "ray_d", "o_cross_d"]
        self.posed_image_keys = self.pose_keys + ["normalized_image"]
        self.color_dim = 3 * (self.config.model.gaussians.sh_degree + 1) ** 2
        self.opacity_dim = 1 * (self.config.model.gaussians.opacity_degree + 1) ** 2
        self._init_tokenizers()
        self.output_gs = config.model.output_gs

        self.stage1 = [
            TransformerBlock(
                config.model.dim1, False, # bias
                config.model.head_dim, config.model.inter_multi,
                config.model.qk_norm)
            for _ in range(config.model.stage1_nlayer)
        ]
        self.stage1 = nn.ModuleList(self.stage1)
        self.stage2 = [
            TransformerBlock(
                config.model.dim2, False, # bias
                config.model.head_dim, config.model.inter_multi,
                config.model.qk_norm)
            for _ in range(config.model.stage2_nlayer)
        ]
        self.stage2 = nn.ModuleList(self.stage2)
        self.stage3 = [
            TransformerBlock(
                config.model.dim3, False, # bias
                config.model.head_dim, config.model.inter_multi,
                config.model.qk_norm)
            for _ in range(config.model.stage3_nlayer)
        ]
        self.stage3 = nn.ModuleList(self.stage3)
        self.apply(_init_weights)

        self.patch_size = config.model.patch_size
        self.num_register_tokens = config.model.num_register_tokens
        # self.group_size = config.model.group_size

        self.register_token_init = nn.Parameter(torch.randn(1, 1, self.num_register_tokens, config.model.dim1))
        nn.init.normal_(self.register_token_init, mean=0.0, std=0.02)

        ### hard-coded Prope attention modules
        if config.training.train_stage == 1:
            self.attention2 = PropeDotProductAttention(
            head_dim=64, patches_x=256, patches_y=128,
            image_width=1024, image_height=512,
            num_register_tokens=self.num_register_tokens)

            self.attention3 = PropeDotProductAttention(
                head_dim=64, patches_x=128, patches_y=64,
                image_width=1024, image_height=512,
                num_register_tokens=self.num_register_tokens)
        # elif config.training.train_stage in [2, 3]:
        # elif config.training.train_stage == 2:
        #     self.attention2 = PropeDotProductAttention(
        #         head_dim=64, patches_x=512, patches_y=256,
        #         image_width=2048, image_height=1024,
        #         num_register_tokens=self.num_register_tokens)

        #     self.attention3 = PropeDotProductAttention(
        #         head_dim=64, patches_x=256, patches_y=128,
        #         image_width=2048, image_height=1024,
        #         num_register_tokens=self.num_register_tokens)
        elif config.training.train_stage == 2:
            self.attention2 = PropeDotProductAttention(
                head_dim=64, patches_x=128, patches_y=64,
                image_width=2048, image_height=1024,
                num_register_tokens=self.num_register_tokens)

            self.attention3 = PropeDotProductAttention(
                head_dim=64, patches_x=64, patches_y=32,
                image_width=2048, image_height=1024,
                num_register_tokens=self.num_register_tokens)
        else:
            raise NotImplementedError

        self.merge_block1 = nn.Conv2d(
            self.dim1, self.dim2, kernel_size=2, stride=2,
            padding=0, bias=True, groups=self.dim1)
        self.resize_block1 = nn.Linear(self.dim1, self.dim2)

        self.merge_block2 = nn.Conv2d(
            self.dim2, self.dim3, kernel_size=2, stride=2,
            padding=0, bias=True, groups=self.dim2)
        self.resize_block2 = nn.Linear(self.dim2, self.dim3)

        self.dpt_head = DPTHead(
            dim_in = [self.dim1, self.dim2, self.dim3],
            features = self.dim3,
            out_channels = [self.dim1, self.dim2, self.dim3],
        )

        self.loss_computer = LossComputer(config)

    def train(self, mode=True):
        """Override the train method to keep the loss computer in eval mode"""
        super().train(mode)
        self.loss_computer.eval()

    def _init_tokenizers(self):
        """Initialize the image and target pose tokenizers, and image token decoder"""
        # Image tokenizer
        self.image_tokenizer = self._create_tokenizer(
            in_channels = self.config.model.in_channels,
            patch_size = self.config.model.patch_size,
            d_model = self.config.model.dim1
        )

        # Image token decoder (decode image tokens into pixels)
        self.gaussian_decoder = nn.Sequential(
            nn.LayerNorm(self.dim3, bias=False),
            nn.Linear(
                self.dim3,
                (self.config.model.patch_size ** 2) * \
                    (3 + self.color_dim + 3 + 4 + self.opacity_dim),
                bias=False))

    def _create_tokenizer(self, in_channels, patch_size, d_model):
        """Helper function to create a tokenizer with given config"""
        tokenizer = nn.Sequential(
            Rearrange(
                "b v c (hh ph) (ww pw) -> b (v hh ww) (ph pw c)",
                ph=patch_size, pw=patch_size),
            nn.Linear(
                in_channels * (patch_size**2), d_model, bias=False),
            nn.LayerNorm(d_model, bias=False))

        return tokenizer

    def render_one(self, xyz, feature, scale, rotation, opacity, test_c2w, test_intr,
               W, H, sh_degree, near_plane, far_plane):
        opacity = opacity.sigmoid().squeeze(-1)
        scale = scale.exp()
        rotation = F.normalize(rotation, p=2, dim=-1)
        test_w2c = test_c2w.float().inverse().unsqueeze(0) # (1, 4, 4)
        # test_w2c = test_c2w.float().inverse()
        test_intr_i = torch.zeros(3, 3).to(test_intr.device)
        test_intr_i[0, 0] = test_intr[0]
        test_intr_i[1, 1] = test_intr[1]
        test_intr_i[0, 2] = test_intr[2]
        test_intr_i[1, 2] = test_intr[3]
        test_intr_i[2, 2] = 1
        test_intr_i = test_intr_i.unsqueeze(0) # (1, 3, 3)
        rendering, _, _ = rasterization(xyz, rotation, scale, opacity, feature,
                                        test_w2c, test_intr_i, W, H, sh_degree=sh_degree,
                                        near_plane=near_plane, far_plane=far_plane,
                                        packed=False,
                                        absgrad=False,
                                        sparse_grad=False,
                                        render_mode="RGB+ED",
                                        backgrounds=torch.ones(1, 3).to(test_intr.device),
                                        rasterize_mode='classic') # (1, H, W, 4)
        return rendering # (1, H, W, 4)

    def normalize_depth(self, depth_map):
        """
        Args:
            depth_map: (B, V, 1, H, W)
        Returns:
            normalized_depth: (B, V, 1, H, W) in range [0, 1]
        """
        B, V, C, H, W = depth_map.shape
        # Flatten H and W to compute min/max: (B, V, H*W).
        depth_flat = depth_map.view(B, V, -1)

        # Compute max and min for each image.
        max_val = depth_flat.max(dim=-1, keepdim=True)[0] # (B, V, 1)
        min_val = depth_flat.min(dim=-1, keepdim=True)[0] # (B, V, 1)

        # Restore dimensions for broadcasting: (B, V, 1, 1, 1).
        max_val = max_val.view(B, V, 1, 1, 1)
        min_val = min_val.view(B, V, 1, 1, 1)

        # Compute the denominator and guard against division by zero.
        denominator = max_val - min_val
        denominator = torch.where(denominator < 1e-6, torch.ones_like(denominator), denominator)

        return (depth_map - min_val) / denominator

    def forward(
        self,
        input_data_dict, # Panorama images.
        target_data_dict, # Perspective images with intrinsics.
    ):
        # Do not autocast during the data processing stage
        with torch.autocast(device_type="cuda", enabled=False), torch.no_grad():
            b_in, v_in, _, h_in, w_in = input_data_dict["input_images"].size() # Panorama images.
            b_target, t_target, _, h_target, w_target = target_data_dict["target_images"].size() # Perspective images.
            # i_fxfycxcy = input_data_dict["fxfycxcy"]
            i_c2w = input_data_dict["input_c2ws"]

            t_fxfycxcy = target_data_dict["target_fxfycxcy"]
            t_c2w = target_data_dict["target_c2ws"]

            ray_o, ray_d = compute_plucmap_pano(i_c2w, h_in, w_in)
            o_cross_d = torch.cross(ray_o, ray_d, dim=2)
            i_normalized_image = input_data_dict["input_images"] * 2.0 - 1.0
            i_raymap_images = torch.concat([ray_o, ray_d, o_cross_d, i_normalized_image], dim=2)

            # Ks = torch.eye(3, dtype=i_c2w.dtype, device=i_c2w.device).unsqueeze(0).unsqueeze(0)
            # Ks = Ks.repeat(b, v, 1, 1).clone()
            # Ks[:, :, 0, 0] = i_fxfycxcy[:, :, 0]
            # Ks[:, :, 1, 1] = i_fxfycxcy[:, :, 1]
            # Ks[:, :, 0, 2] = i_fxfycxcy[:, :, 2]
            # Ks[:, :, 1, 2] = i_fxfycxcy[:, :, 3]
            # Ks[:, :, 2, 2] = 1.0

            i_w2c = torch.inverse(i_c2w)


        register_tokens = self.register_token_init.repeat(b_in, v_in, 1, 1)

        x = self.image_tokenizer(i_raymap_images)
        x = rearrange(x, "b (v l) d -> b v l d", v=v_in)
        x = torch.cat([register_tokens, x], dim=2)  # Add register tokens
        x = rearrange(x, "b v l d -> (b v) l d")
        x = self.run_stage1(x, None)
        r_tokens1, i_tokens1_prev = x[:, :self.num_register_tokens], x[:, self.num_register_tokens:]
        r_tokens1 = self.resize_block1(r_tokens1)
        hh1 = h_in // self.patch_size
        ww1 = w_in // self.patch_size
        i_tokens1 = rearrange(
            i_tokens1_prev, "b (hh ww) d -> b d hh ww",
            hh=hh1, ww=ww1)
        i_tokens1 = self.merge_block1(i_tokens1)
        i_tokens1 = rearrange(
            i_tokens1, "b d hh ww -> b (hh ww) d",
            hh=hh1//2, ww=ww1//2)
        x = torch.cat([r_tokens1, i_tokens1], dim=1)
        x = rearrange(x, "(b v) l d -> b (v l) d", v=v_in)

        info_stage2 = {
            "num_input_views": v_in,
            "w2c": i_w2c,
            "attn2": self.attention2,
            "input_room_ids": input_data_dict.get("input_room_ids"),
        }
        x = self.run_stage2(x, info_stage2)
        r_tokens2, i_tokens2_prev = x[:, :self.num_register_tokens], x[:, self.num_register_tokens:]
        r_tokens2 = self.resize_block2(r_tokens2)
        hh2 = hh1 // 2
        ww2 = ww1 // 2
        i_tokens2 = rearrange(
            i_tokens2_prev, "b (hh ww) d -> b d hh ww",
            hh=hh2, ww=ww2)
        i_tokens2 = self.merge_block2(i_tokens2)
        i_tokens2 = rearrange(
            i_tokens2, "b d hh ww -> b (hh ww) d",
            hh=hh2//2, ww=ww2//2)
        x = torch.cat([r_tokens2, i_tokens2], dim=1)
        x = rearrange(x, "(b v) l d -> b (v l) d", v=v_in)

        info_stage3 = {
            "num_input_views": v_in,
            "attn3": self.attention3,
            "w2c": i_w2c,
            "input_room_ids": input_data_dict.get("input_room_ids"),
        }
        x = self.run_stage3(x, info_stage3)
        i_tokens3_prev = x[:, self.num_register_tokens:]

        output_tokens = self.dpt_head(
            [i_tokens1_prev, i_tokens2_prev, i_tokens3_prev], [h_in, w_in], self.patch_size
        )
        output_tokens = rearrange(output_tokens, "(b v) l d -> b (v l) d", v=v_in)
        gaussians = self.gaussian_decoder(output_tokens)
        gaussians = rearrange(
            gaussians, "b (v hh ww) (ph pw d) -> b (v hh ph ww pw) d", v=v_in,
            hh=h_in // self.config.model.patch_size,
            ww=w_in // self.config.model.patch_size,
            ph=self.config.model.patch_size,
            pw=self.config.model.patch_size)
        xyz, feature, scale, rotation, opacity_sh = torch.split(gaussians, [3, self.color_dim, 3, 4, self.opacity_dim], dim=-1)
        xyz = xyz.float() # (B, V*H*W, 3)
        feature = feature.float() # (B, V*H*W, 3 * (sh_degree + 1) ** 2)
        scale = scale.float() # (B, V*H*W, 3)
        rotation = rotation.float() # (B, V*H*W, 4)
        opacity_sh = opacity_sh.float() # (B, V*H*W, 1 * (opacity_degree + 1) ** 2)
        with torch.autocast(device_type="cuda", enabled=False):
            rayo_gs, rayd_gs = compute_rays_pano(i_c2w, h_in, w_in)
            scale = (scale + self.config.model.gaussians.scale_bias).clamp(max = self.config.model.gaussians.scale_max)
            # opacity bias only for the sh0 component
            opacity_sh = opacity_sh + self.config.model.gaussians.opacity_bias
            feature = rearrange(feature, "b n (c d) -> b n d c", c=3).contiguous()
            opacity_mean = opacity_sh.mean(dim=2, keepdim=True)
            opacity_precompute = opacity_mean.repeat([1, 1, t_target]).permute(0,2,1).unsqueeze(3).contiguous()
            inv_min_dist = 1.0 / self.config.model.gaussians.max_dist
            inv_max_dist = 1.0 / self.config.model.gaussians.min_dist
            # The model predicts inverse depth; convert it to metric depth here.
            inv_dist = (xyz.mean(dim=-1, keepdim=True) - 3.0).sigmoid() * (inv_max_dist - inv_min_dist) + inv_min_dist # (B, V*H*W, 1)
            dist = 1.0 / (inv_dist + 1e-6)
            xyz = dist * rayd_gs + rayo_gs

        gaussians = {
            "xyz": xyz, # shape: torch.Size([1, 3932160, 3])
            "feature": feature, # shape: torch.Size([1, 3932160, 4, 3])
            "scale": scale, # shape: torch.Size([1, 3932160, 3])
            "rotation": rotation, # shape: torch.Size([1, 3932160, 4])
            "opacity": opacity_precompute, # torch.Size([1, 8, 3932160, 1])
        }

        with torch.autocast(device_type="cuda", enabled=False):
            # Rasterization
            renderings_raw = GaussianRenderer.apply(
                gaussians["xyz"],
                gaussians["feature"],
                gaussians["scale"],
                gaussians["rotation"],
                gaussians["opacity"],
                t_c2w,
                t_fxfycxcy,
                w_target, h_target,
                self.config.model.gaussians.sh_degree,
                self.config.model.gaussians.near_plane,
                self.config.model.gaussians.far_plane,
            ) # (B, V, H, W, 4)

        renderings_raw = renderings_raw.permute(0, 1, 4, 2, 3).contiguous() # (b_target, t_target, 4, h_target, w_target)

        # Split RGB and depth.
        renderings_rgb = renderings_raw[:, :, :3, :, :]   # (b_target, t_target, 3, h_target, w_target) # Rendered perspective RGB.
        renderings_depth = renderings_raw[:, :, 3:, :, :] # (b_target, t_target, 1, h_target, w_target) # Rendered perspective depth.
        # renderings_depth_norm = self.normalize_depth(renderings_depth)
        renderings_depth3d = dist.reshape(b_in, v_in, 1, h_in, w_in) # Panorama Gaussian spherical distance from its source camera.

        # Compute depth loss.
        batch_size = renderings_depth.shape[0]
        view_size = renderings_depth.shape[1]
        batch_losses = []
        batch_abs_depth_distance = []
        valid_views = 0

        # Supervise input point-cloud depth.
        for b_idx in range(renderings_depth3d.shape[0]):
            # Extract the current batch slice (V, 1, H, W).
            curr_pred = renderings_depth3d[b_idx]
            curr_gt = input_data_dict["input_depths"][b_idx]
            curr_mask = input_data_dict["input_depths_mask"][b_idx]
            # Use the mask to extract valid pixels as a flat vector.
            # The resulting shape is (N_valid_pixels,).
            pred_valid = curr_pred[curr_mask]
            gt_valid = curr_gt[curr_mask].detach()

            # Boundary check: skip if the current batch has no valid pixels.
            if gt_valid.numel() == 0:
                loss_b = renderings_depth.sum() * 0.0
                batch_losses.append(loss_b)
                continue

            # Filter out out-of-range GT values as an additional safeguard.
            valid_range_mask = (gt_valid > self.config.model.gaussians.min_dist) & (gt_valid < self.config.model.gaussians.max_dist + 1.0) # Slightly relax the upper bound.
            if valid_range_mask.sum() == 0:
                loss_b = renderings_depth.sum() * 0.0
                batch_losses.append(loss_b)
                continue

            pred_valid = pred_valid[valid_range_mask]
            gt_valid = gt_valid[valid_range_mask]

            depth_l1_log_loss = F.l1_loss(torch.log(pred_valid + 1.0), torch.log(gt_valid + 1.0))
            log_diff = torch.log(pred_valid + 1e-6) - torch.log(gt_valid + 1e-6)
            var_term = (log_diff ** 2).mean() - 0.85 * (log_diff.mean() ** 2)
            scale_invariant_loss = torch.sqrt(var_term + 1e-8) * 0.1

            loss_b = depth_l1_log_loss + scale_invariant_loss
            batch_losses.append(loss_b)
            valid_views += 1

            batch_abs_depth_distance.append(torch.abs(pred_valid - gt_valid).mean())

        # Aggregate losses.
        if valid_views == 0:
            # If all batches are invalid, return a differentiable zero loss.
            final_depth_loss = renderings_depth.sum() * 0.0
            final_abs_depth = renderings_depth.sum() * 0.0
        else:
            # Stack the list into a tensor and average it.
            final_depth_loss = torch.stack(batch_losses).mean()
            final_abs_depth = torch.stack(batch_abs_depth_distance).mean()

        # Supervise rendered perspective depth.
        batch_losses_target = []
        valid_views = 0
        for b_idx in range(renderings_depth.shape[0]):
            # Extract the current batch slice (V, 1, H, W).
            curr_pred = renderings_depth[b_idx]
            curr_gt = target_data_dict["target_depths"][b_idx]
            curr_mask = target_data_dict["target_depths_mask"][b_idx]

            # Use the mask to extract valid pixels as a flat vector.
            # The resulting shape is (N_valid_pixels,).
            pred_valid = curr_pred[curr_mask]
            gt_valid = curr_gt[curr_mask].detach()

            # Boundary check: skip if the current batch has no valid pixels.
            if gt_valid.numel() == 0:
                loss_b = renderings_depth.sum() * 0.0
                batch_losses_target.append(loss_b)
                continue

            # Filter out out-of-range GT values as an additional safeguard.
            valid_range_mask = (gt_valid > self.config.model.gaussians.min_dist) & (gt_valid < self.config.model.gaussians.max_dist + 1.0) # Slightly relax the upper bound.
            if valid_range_mask.sum() == 0:
                loss_b = renderings_depth.sum() * 0.0
                batch_losses_target.append(loss_b)
                continue

            pred_valid = pred_valid[valid_range_mask]
            gt_valid = gt_valid[valid_range_mask]

            depth_l1_log_loss = F.l1_loss(torch.log(pred_valid + 1.0), torch.log(gt_valid + 1.0))
            log_diff = torch.log(pred_valid + 1e-6) - torch.log(gt_valid + 1e-6)
            var_term = (log_diff ** 2).mean() - 0.85 * (log_diff.mean() ** 2)
            scale_invariant_loss = torch.sqrt(var_term + 1e-8) * 0.1

            loss_b_target = depth_l1_log_loss + scale_invariant_loss
            batch_losses_target.append(loss_b_target)
            valid_views += 1

        # Aggregate losses.
        if valid_views == 0:
            # If all batches are invalid, return a differentiable zero loss.
            final_depth_loss_target = renderings_depth.sum() * 0.0
        else:
            # Stack the list into a tensor and average it.
            final_depth_loss_target = torch.stack(batch_losses_target).mean()

        loss_metrics = self.loss_computer(
            renderings_rgb,
            target_data_dict["target_images"]
        )
        with torch.autocast(device_type="cuda", enabled=False):
            opacity_random = opacity_sh.sigmoid().mean()

        loss_metrics["opacity_loss"] = opacity_random * 0.0
        loss_metrics["depth_loss"] = final_depth_loss * self.config.training.get("depth_loss_weight", 1.0)
        loss_metrics["depth_render_loss"] = final_depth_loss_target * self.config.training.get("depth_render_loss_weight", 0.0)
        loss_metrics["loss"] = loss_metrics["loss"] + loss_metrics["opacity_loss"] + loss_metrics["depth_loss"] + loss_metrics["depth_render_loss"]
        loss_metrics["final_abs_depth"] = final_abs_depth

        if self.output_gs:
            result = edict(
                input=input_data_dict,
                target=target_data_dict,
                loss_metrics=loss_metrics,
                render=renderings_rgb,
                depth=renderings_depth, # Rendered depth map.
                depth_dist=renderings_depth3d, # Positional depth map.
                gs_params=gaussians,
                )
        else:
            result = edict(
                input=input_data_dict,
                target=target_data_dict,
                loss_metrics=loss_metrics,
                render=renderings_rgb,
                depth=renderings_depth, # Depth map.
                depth_dist=renderings_depth3d, # Positional depth map.
                )

        return result

    def run_stage1(self, x, info):
        for i in range(len(self.stage1)):
            x = torch.utils.checkpoint.checkpoint(
                self.stage1[i], x, False, 1, info, use_reentrant=False)
        return x

    def run_stage2(self, x, info):
        # g = self.group_size
        v = info["num_input_views"]
        for i in range(len(self.stage2)):
            if i % 2 == 0:
                # x = rearrange(
                #     x, "(b g) (v l) d -> (b g v) l d", g=v//g, v=g)
                x = rearrange(x, "b (v l) d -> (b v) l d", v=v)
                x = torch.utils.checkpoint.checkpoint(
                    self.stage2[i], x, False, 2, info, use_reentrant=False)
                # x = rearrange(
                #     x, "(b g v) l d -> (b g) (v l) d", g=v//g, v=g)
                x = rearrange(x, "(b v) l d -> b (v l) d", v=v)
            else:
                x = torch.utils.checkpoint.checkpoint(
                    self.stage2[i], x, True, 2, info, use_reentrant=False)
        # return rearrange(x, "(b g) (v l) d -> (b g v) l d", g=v//g, v=g)
        return rearrange(x, "b (v l) d -> (b v) l d", v=v)

    def run_stage3(self, x, info):
        v = info["num_input_views"]
        for i in range(len(self.stage3)):
            if i % 2 == 0:
                x = rearrange(x, "b (v l) d -> (b v) l d", v=v)
                x = torch.utils.checkpoint.checkpoint(
                    self.stage3[i], x, False, 3, info, use_reentrant=False)
                x = rearrange(x, "(b v) l d -> b (v l) d", v=v)
            else:
                x = torch.utils.checkpoint.checkpoint(
                    self.stage3[i], x, True, 3, info, use_reentrant=False)
        return rearrange(x, "b (v l) d -> (b v) l d", v=v)

    def save_input_video(self, input_intr, input_c2ws, gaussian_dict, H, W, save_path, insert_frame_num = 16):
        """
        Interpolate input frames and save rendered video
        input_intr: (V, 4), (fx, fy, cx, cy)
        input_c2ws: (V, 4, 4)
        """
        import cv2
        from camera_utils import get_interpolated_poses_many
        import subprocess
        V = input_intr.shape[0]
        device = input_intr.device
        input_intr = input_intr.detach().cpu().float()
        input_c2ws = input_c2ws.detach().cpu().float()

        input_intr_mat = torch.zeros((V, 3, 3))
        input_intr_mat[:, 0, 0] = input_intr[:, 0]
        input_intr_mat[:, 1, 1] = input_intr[:, 1]
        input_intr_mat[:, 0, 2] = input_intr[:, 2]
        input_intr_mat[:, 1, 2] = input_intr[:, 3]
        input_c2ws = torch.cat([input_c2ws, input_c2ws[:1]], dim=0) # wrap around
        input_intr_mat = torch.cat([input_intr_mat, input_intr_mat[:1]], dim=0) # wrap around
        c2ws, intr_mat, _ = get_interpolated_poses_many(input_c2ws[:, :3, :4], input_intr_mat, steps_per_transition = insert_frame_num)
        V = c2ws.shape[0]
        c2ws_mat = torch.eye(4).unsqueeze(0).repeat(V, 1, 1)
        c2ws_mat[:, :3, :4] = c2ws
        intr_fxfycxcy = torch.zeros(V, 4)
        intr_fxfycxcy[:, 0] = intr_mat[:, 0, 0]
        intr_fxfycxcy[:, 1] = intr_mat[:, 1, 1]
        intr_fxfycxcy[:, 2] = intr_mat[:, 0, 2]
        intr_fxfycxcy[:, 3] = intr_mat[:, 1, 2]
        c2ws_mat = c2ws_mat.to(device)
        intr_fxfycxcy = intr_fxfycxcy.to(device)

        xyz = gaussian_dict["xyz"].detach().float().to(device) # (N, 3)
        feature = gaussian_dict["feature"].detach().float().to(device) # (N, (sh_degree+1)**2, 3)
        scale = gaussian_dict["scale"].detach().float().to(device) # (N, 3)
        rotation = gaussian_dict["rotation"].detach().float().to(device) # (N, 4)
        opacity = gaussian_dict["opacity"].detach().float().to(device)

        renderings = []
        with torch.autocast(enabled=False, device_type="cuda"):
            for i in range(V):
                dir = xyz - c2ws_mat[i:i+1, :3, 3][None, ...] # (1, N, 3)
                opacity_i = _spherical_harmonics(
                    self.config.model.gaussians.opacity_degree,
                    dir, opacity[None, ...])[0] # (N, 1)
                rendering = self.render_one(xyz, feature, scale, rotation, opacity_i,
                                            c2ws_mat[i], intr_fxfycxcy[i], W, H,
                                            self.config.model.gaussians.sh_degree,
                                            self.config.model.gaussians.near_plane,
                                            self.config.model.gaussians.far_plane)
                rendering = rendering[..., :3].squeeze(0).clamp(0, 1).cpu().numpy() # (H, W, 3)
                rendering = (rendering * 255).astype(np.uint8)
                rendering = cv2.cvtColor(rendering, cv2.COLOR_RGB2BGR)
                renderings.append(rendering)
        tmp_save_path = save_path.replace(".mp4", "_tmp.mp4")
        video_writer = cv2.VideoWriter(tmp_save_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (W, H))
        for r in renderings:
            video_writer.write(r)
        video_writer.release()
        subprocess.run(f"ffmpeg -y -i {tmp_save_path} -vcodec libx264 -f mp4 {save_path} -loglevel quiet", shell=True)
        os.remove(tmp_save_path)

    @torch.no_grad()
    def load_ckpt(self, load_path):
        if os.path.isdir(load_path):
            ckpt_names = [file_name for file_name in os.listdir(load_path) if file_name.endswith(".pt")]
            ckpt_names = sorted(ckpt_names, key=lambda x: x)
            ckpt_paths = [os.path.join(load_path, ckpt_name) for ckpt_name in ckpt_names]
        else:
            ckpt_paths = [load_path]
        try:
            checkpoint = torch.load(ckpt_paths[-1], map_location="cpu", weights_only=True)
        except:
            traceback.print_exc()
            print(f"Failed to load {ckpt_paths[-1]}")
            return None

        self.load_state_dict(checkpoint["ema"], strict=False)
        return 0
