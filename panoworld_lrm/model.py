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
    compute_rays_pano,
    compute_plucmap_pano,
)
from .dpt_head import DPTHead
from .prope_custom import PropeDotProductAttention


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
        try:
            test_w2c = test_c2w.float().inverse().unsqueeze(0)
        except RuntimeError:
            print(f"Matrix is not invertible, test_c2w: {test_c2w}")
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
            # Initialize a 4-channel tensor to store RGB(3) + Depth(1)
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


class PanoWorldLRM(nn.Module):
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
        elif config.training.train_stage == 2:
            self.attention2 = PropeDotProductAttention(
                head_dim=64, patches_x=512, patches_y=256,
                image_width=2048, image_height=1024,
                num_register_tokens=self.num_register_tokens)

            self.attention3 = PropeDotProductAttention(
                head_dim=64, patches_x=256, patches_y=128,
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


    def train(self, mode=True):
        """Override the train method to keep the loss computer in eval mode"""
        super().train(mode)

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

    def forward(
        self,
        input_data_dict, # Panorama inputs
        target_data_dict, # Perspective targets with intrinsics
    ):
        # Do not autocast during the data processing stage
        with torch.autocast(device_type="cuda", enabled=False), torch.no_grad():
            b_in, v_in, _, h_in, w_in = input_data_dict["input_images"].size() # Panorama inputs
            t_c2w = target_data_dict["target_c2ws"]
            b_target = b_in
            t_target = t_c2w[0].shape[0]
            h_target = self.config.data.resize_h
            w_target = self.config.data.resize_h
            i_c2w = input_data_dict["input_c2ws"]
            t_fxfycxcy = target_data_dict["target_fxfycxcy"]

            ray_o, ray_d = compute_plucmap_pano(i_c2w, h_in, w_in)
            o_cross_d = torch.cross(ray_o, ray_d, dim=2)
            i_normalized_image = input_data_dict["input_images"] * 2.0 - 1.0
            i_raymap_images = torch.concat([ray_o, ray_d, o_cross_d, i_normalized_image], dim=2)

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
        xyz = xyz.float()
        feature = feature.float()
        scale = scale.float()
        rotation = rotation.float()
        opacity_sh = opacity_sh.float()
        with torch.autocast(device_type="cuda", enabled=False):
            rayo_gs, rayd_gs = compute_rays_pano(i_c2w, h_in, w_in)
            scale = (scale + self.config.model.gaussians.scale_bias).clamp(max = self.config.model.gaussians.scale_max)
            opacity_sh = opacity_sh + self.config.model.gaussians.opacity_bias
            feature = rearrange(feature, "b n (c d) -> b n d c", c=3).contiguous()
            opacity_mean = opacity_sh.mean(dim=2, keepdim=True)
            opacity_precompute = opacity_mean.repeat([1, 1, t_target]).permute(0,2,1).unsqueeze(3).contiguous()
            inv_min_dist = 1.0 / self.config.model.gaussians.max_dist
            inv_max_dist = 1.0 / self.config.model.gaussians.min_dist
            inv_dist = (xyz.mean(dim=-1, keepdim=True) - 3.0).sigmoid() * (inv_max_dist - inv_min_dist) + inv_min_dist
            dist = 1.0 / (inv_dist + 1e-6)
            xyz = dist * rayd_gs + rayo_gs

        gaussians = {
            "xyz": xyz,
            "feature": feature,
            "scale": scale,
            "rotation": rotation,
            "opacity": opacity_precompute,
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

        # Split RGB and depth outputs
        renderings_rgb = renderings_raw[:, :, :3, :, :]   # (b_target, t_target, 3, h_target, w_target) # Rendered perspective RGB
        renderings_depth = renderings_raw[:, :, 3:, :, :] # (b_target, t_target, 1, h_target, w_target) # Rendered perspective depth
        renderings_depth3d = dist.reshape(b_in, v_in, 1, h_in, w_in) # Per-panorama Gaussian spherical distance from the source camera

        result = edict(
            input=input_data_dict,
            target=target_data_dict,
            render=renderings_rgb,
            depth=renderings_depth,
            depth_dist=renderings_depth3d,
            gs_params=gaussians,
            )

        return result

    def run_stage1(self, x, info):
        for i in range(len(self.stage1)):
            x = self._run_transformer_block(self.stage1[i], x, False, 1, info)
        return x

    def run_stage2(self, x, info):
        v = info["num_input_views"]
        for i in range(len(self.stage2)):
            if i % 2 == 0:
                x = rearrange(x, "b (v l) d -> (b v) l d", v=v)
                x = self._run_transformer_block(self.stage2[i], x, False, 2, info)
                x = rearrange(x, "(b v) l d -> b (v l) d", v=v)
            else:
                x = self._run_transformer_block(self.stage2[i], x, True, 2, info)
        return rearrange(x, "b (v l) d -> (b v) l d", v=v)

    def run_stage3(self, x, info):
        v = info["num_input_views"]
        for i in range(len(self.stage3)):
            if i % 2 == 0:
                x = rearrange(x, "b (v l) d -> (b v) l d", v=v)
                x = self._run_transformer_block(self.stage3[i], x, False, 3, info)
                x = rearrange(x, "(b v) l d -> b (v l) d", v=v)
            else:
                x = self._run_transformer_block(self.stage3[i], x, True, 3, info)
        return rearrange(x, "b (v l) d -> (b v) l d", v=v)

    def _run_transformer_block(self, block, x, prope, stage, info):
        if torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(
                block, x, prope, stage, info, use_reentrant=False
            )
        return block(x, prope, stage, info)

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
