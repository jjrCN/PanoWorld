# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

import lpips
import torch.nn as nn
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from torchvision.models import vgg19
import scipy.io
import os
from pathlib import Path


# the perception loss code is modified from https://github.com/zhengqili/Crowdsampling-the-Plenoptic-Function/blob/f5216f312cf82d77f8d20454b5eeb3930324630a/models/networks.py#L1478
# and some parts are based on https://github.com/arthurhero/Long-LRM/blob/main/model/loss.py
class PerceptualLoss(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.vgg = self._build_vgg()
        self._load_weights()
        self._setup_feature_blocks()

    def _build_vgg(self):
        """Create VGG model with average pooling instead of max pooling."""
        model = vgg19()
        # Replace max pooling with average pooling
        for i, layer in enumerate(model.features):
            if isinstance(layer, nn.MaxPool2d):
                model.features[i] = nn.AvgPool2d(kernel_size=2, stride=2)

        return model.to(self.device).eval()

    def _load_weights(self):
        """Load pre-trained VGG weights. """
        weight_file = Path("./imagenet-vgg-verydeep-19.mat")
        weight_file.parent.mkdir(exist_ok=True, parents=True)

        if not weight_file.exists():
            os.system(f'wget https://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat -O {weight_file}')

        # Load MatConvNet weights
        vgg_data = scipy.io.loadmat(weight_file)
        vgg_layers = vgg_data["layers"][0]

        # Layer indices and filter sizes
        layer_indices = [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]
        filter_sizes = [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]

        # Transfer weights to PyTorch model
        with torch.no_grad():
            for i, layer_idx in enumerate(layer_indices):
                # Set weights
                weights = torch.from_numpy(vgg_layers[layer_idx][0][0][2][0][0]).permute(3, 2, 0, 1)
                self.vgg.features[layer_idx].weight = nn.Parameter(weights, requires_grad=False)

                # Set biases
                biases = torch.from_numpy(vgg_layers[layer_idx][0][0][2][0][1]).view(filter_sizes[i])
                self.vgg.features[layer_idx].bias = nn.Parameter(biases, requires_grad=False)

    def _setup_feature_blocks(self):
        """Create feature extraction blocks at different network depths."""
        output_indices = [0, 4, 9, 14, 23, 32]
        self.blocks = nn.ModuleList()

        # Create sequential blocks
        for i in range(len(output_indices) - 1):
            block = nn.Sequential(*list(self.vgg.features[output_indices[i]:output_indices[i+1]]))
            self.blocks.append(block.to(self.device).eval())

        # Freeze all parameters
        for param in self.vgg.parameters():
            param.requires_grad = False

    def _extract_features(self, x):
        """Extract features from each block."""
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features

    def _preprocess_images(self, images):
        """Convert images to VGG input format."""
        # VGG mean values for ImageNet
        mean = torch.tensor([123.6800, 116.7790, 103.9390]).reshape(1, 3, 1, 1).to(images.device)
        return images * 255.0 - mean

    @staticmethod
    def _compute_error(real, fake):
        return torch.mean(torch.abs(real - fake))

    def forward(self, pred_img, target_img):
        """Compute perceptual loss between prediction and target."""
        # Preprocess images
        target_img_p = self._preprocess_images(target_img)
        pred_img_p = self._preprocess_images(pred_img)

        # Extract features
        target_features = self._extract_features(target_img_p)
        pred_features = self._extract_features(pred_img_p)

        # Pixel-level error
        e0 = self._compute_error(target_img_p, pred_img_p)

        # Feature-level errors with scaling factors
        e1 = self._compute_error(target_features[0], pred_features[0]) / 2.6
        e2 = self._compute_error(target_features[1], pred_features[1]) / 4.8
        e3 = self._compute_error(target_features[2], pred_features[2]) / 3.7
        e4 = self._compute_error(target_features[3], pred_features[3]) / 5.6
        e5 = self._compute_error(target_features[4], pred_features[4]) * 10 / 1.5

        # Combine all errors and normalize
        total_loss = (e0 + e1 + e2 + e3 + e4 + e5) / 255.0

        return total_loss

class LossComputer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if self.config.training.lpips_loss_weight > 0.0:
            # avoid multiple GPUs from downloading the same LPIPS model multiple times
            if torch.distributed.get_rank() == 0:
                self.lpips_loss_module = self._init_frozen_module(lpips.LPIPS(net="vgg"))
            torch.distributed.barrier()
            if torch.distributed.get_rank() != 0:
                self.lpips_loss_module = self._init_frozen_module(lpips.LPIPS(net="vgg"))
        if self.config.training.perceptual_loss_weight > 0.0:
            self.perceptual_loss_module = self._init_frozen_module(PerceptualLoss())

    def _init_frozen_module(self, module):
        """Helper method to initialize and freeze a module's parameters."""
        module.eval()
        for param in module.parameters():
            param.requires_grad = False
        return module

    def forward(
        self,
        rendering,
        target,
    ):
        """
        Calculate various losses between rendering and target images.

        Args:
            rendering: [b, v, 3, h, w], value range [0, 1]
            target: [b, v, 3, h, w], value range [0, 1]

        Returns:
            Dictionary of loss metrics
        """
        b, v, _, h, w = rendering.size()
        rendering = rendering.reshape(b * v, -1, h, w)
        target = target.reshape(b * v, -1, h, w)

        # Handle alpha channel if present
        if target.size(1) == 4:
            target, _ = target.split([3, 1], dim=1)

        l2_loss = torch.tensor(1e-8).to(rendering.device)
        if self.config.training.l2_loss_weight > 0.0:
            l2_loss = F.mse_loss(rendering, target)

        psnr = -10.0 * torch.log10(l2_loss)

        lpips_loss = torch.tensor(0.0).to(l2_loss.device)
        if self.config.training.lpips_loss_weight > 0.0:
            # Scale from [0,1] to [-1,1] as required by LPIPS
            lpips_loss = self.lpips_loss_module(
                rendering * 2.0 - 1.0, target * 2.0 - 1.0
            ).mean()

        perceptual_loss = torch.tensor(0.0).to(l2_loss.device)
        if self.config.training.perceptual_loss_weight > 0.0:
            perceptual_loss = self.perceptual_loss_module(rendering, target)


        loss = (
            self.config.training.l2_loss_weight * l2_loss
            + self.config.training.lpips_loss_weight * lpips_loss
            + self.config.training.perceptual_loss_weight * perceptual_loss
        )


        loss_metrics = edict(
            loss=loss,
            l2_loss=l2_loss * self.config.training.l2_loss_weight,
            psnr=psnr,
            lpips_loss=lpips_loss * self.config.training.lpips_loss_weight,
            perceptual_loss=perceptual_loss * self.config.training.perceptual_loss_weight,
            norm_perceptual_loss=perceptual_loss / l2_loss,
            norm_lpips_loss=lpips_loss / l2_loss
        )
        return loss_metrics