#!/usr/bin/env python
# coding=utf-8
# v0.5.5: Curriculum Learning with Mixed Training in Stage 2 - Based on v0.5.4
# Key changes: Enhanced curriculum learning with batch-level mixing:
#   Stage 1 (Simple): Control image only conditioning (0 ~ stage1_steps)
#   Stage 2 (Mixed): Gradually increasing style image usage with batch-level mixing
#     - Progressive increase: Start with more control-only batches, gradually add more style batches
#     - Batch-level: Each batch uniformly uses either control-only or control+style
#     - Default ratio: 20% control-only, 80% control+style (gradually reached)
# Switching mechanism: Based on training steps (global_step)
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
"""Train the PanoWorld 2D Generator LoRA from a public JSONL manifest.

This is the portable release of the training implementation used for the
checkpoint-5000 model.  Internal data generation and cluster paths have been
replaced with an explicit four-image manifest contract.
"""
import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path

import torch
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo
from peft.utils import get_peft_model_state_dict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import diffusers
from diffusers import QwenImageEditPipeline
from diffusers.training_utils import free_memory
from diffusers.utils import is_wandb_available
from diffusers.utils.import_utils import is_torch_npu_available

from .data import BucketDataset, PanoWorldTrainingDataset, collate_training_batch
from .model import (
    compute_stage2_style_prob,
    create_lr_scheduler,
    create_optimizer,
    load_training_models,
    train_one_batch,
)

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.36.0.dev0")

logger = get_logger(__name__)

if is_torch_npu_available():
    torch.npu.config.allow_internal_format = False


# Constants from QwenImageEditPlusPipeline
CONDITION_IMAGE_SIZE = 384 * 384
VAE_IMAGE_SIZE = 1024 * 1024


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="Qwen/Qwen-Image-Edit-2509",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--bnb_quantization_config_path",
        type=str,
        default=None,
        help="Quantization config in a JSON file that will be used to define the bitsandbytes quant config of the DiT.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--train_manifest",
        type=str,
        required=True,
        help="JSONL manifest with target_panorama, geometric_proxy, visual_memory, and style_reference paths.",
    )
    parser.add_argument("--data_root", type=str, default=None, help="Optional base directory for relative manifest paths.")
    parser.add_argument("--target_width", type=int, default=None, help="Optional padded target width for smoke tests.")
    parser.add_argument("--target_height", type=int, default=None, help="Optional target height for smoke tests.")
    parser.add_argument(
        "--inputs_unpadded",
        action="store_true",
        default=True,
        help="Apply 12.5%% circular padding to target_panorama/geometric_proxy/visual_memory images while loading. Enabled by default.",
    )
    parser.add_argument(
        "--inputs_padded",
        dest="inputs_unpadded",
        action="store_false",
        help="Use when manifest images are already circularly padded to the training width.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help=("A folder containing the training data. "),
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image. By "
        "default, the standard Image Dataset maps out 'file_name' "
        "to 'image'.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default=None,
        help="The column of the dataset containing the instance prompt for each image",
    )

    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")

    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=False,
        help="The prompt with identifier specifying the instance, e.g. 'photo of a TOK dog', 'in the style of TOK'",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length to use with the Qwen2.5 VL as text encoder.",
    )

    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )

    parser.add_argument(
        "--skip_final_inference",
        default=False,
        action="store_true",
        help="Whether to skip the final inference step with loaded lora weights upon training completion. This will run intermediate validation inference if `validation_prompt` is provided. Specify to reduce memory.",
    )

    parser.add_argument(
        "--final_validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during a final validation to verify that the model is learning. Ignored if `--validation_prompt` is provided.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=64,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=64,
        help="LoRA alpha to be used for additional scaling.",
    )
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="Dropout probability for LoRA layers")

    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/panoworld_2d_generator",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=1000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=10,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=150, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodigy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--lora_layers",
        type=str,
        default=None,
        help=(
            'The transformer modules to apply LoRA training on. Please specify the layers in a comma separated. E.g. - "to_k,to_q,to_v" will result in lora training of attention layers only'
        ),
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--cache_latents",
        action="store_true",
        default=False,
        help="Cache the VAE latents",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--upcast_before_saving",
        action="store_true",
        default=False,
        help=(
            "Whether to upcast the trained transformer layers to float32 before saving (at the end of training). "
            "Defaults to precision dtype used for training to save memory"
        ),
    )
    parser.add_argument(
        "--offload",
        action="store_true",
        help="Whether to offload the VAE and the text encoder to CPU when they are not used.",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    # Bucketed training arguments.
    parser.add_argument(
        "--disable_bucket_training",
        action="store_true",
        help="Disable bucket training for different resolutions",
    )

    # v0.5.1: Target-control swapping for bidirectional generation (DISABLED in v0.5.4)
    parser.add_argument(
        "--target_control_swap_prob",
        type=float,
        default=0.0,
        help="Probability of swapping target and control (DISABLED in v0.5.4 curriculum learning)",
    )

    # v0.5.4/v0.5.5: Curriculum Learning parameters
    parser.add_argument(
        "--curriculum_learning",
        action="store_true",
        default=True,
        help="Enable curriculum learning (v0.5.5): Stage 1 (control only) → Stage 2 (mixed control+style)",
    )
    parser.add_argument(
        "--curriculum_stage1_steps",
        type=int,
        default=10,
        help="Number of training steps for Stage 1 (control only). Typically 2-3 epochs. (v0.5.5)",
    )
    parser.add_argument(
        "--curriculum_log_interval",
        type=int,
        default=10,
        help="Log current curriculum stage every N steps (v0.5.5)",
    )

    # v0.5.5: Stage 2 mixed training parameters
    parser.add_argument(
        "--stage2_style_start_prob",
        type=float,
        default=0,
        help="Initial probability of using style image at the beginning of Stage 2 (v0.5.5)",
    )
    parser.add_argument(
        "--stage2_style_end_prob",
        type=float,
        default=0.8,
        help="Final probability of using style image at the end of training (v0.5.5). Default 0.8 means 80%% style, 20%% control-only.",
    )
    parser.add_argument(
        "--stage2_style_warmup_steps",
        type=int,
        default=10,
        help="Number of steps to gradually increase style image usage in Stage 2 (v0.5.5)",
    )
    parser.add_argument(
        "--stage2_mixing_mode",
        type=str,
        default="linear",
        choices=["linear", "cosine", "exp"],
        help="How to increase style image probability: linear, cosine, or exponential (v0.5.5)",
    )

    # v0.5.0/v0.5.3_ref: Control Image augmentation parameters
    parser.add_argument(
        "--control_latent_dropout_prob",
        type=float,
        default=0.05,
        help="Probability of completely dropping control latents (v0.5.0)",
    )
    parser.add_argument(
        "--control_latent_spatial_mask_prob",
        type=float,
        default=0.0,
        help="Probability of applying spatial masking to control latents (v0.5.0)",
    )
    parser.add_argument(
        "--control_latent_spatial_mask_max_ratio",
        type=float,
        default=0.0,
        help="Maximum ratio of spatial regions to mask in control latents (v0.5.0)",
    )
    parser.add_argument(
        "--control_latent_noise_scale",
        type=float,
        default=0.0,
        help="Scale of gaussian noise to add to control latents (v0.5.0)",
    )

    # v0.5.3_ref: Style Image (style reference) augmentation parameters (from v0.4.1)
    parser.add_argument(
        "--style_image_flip_prob",
        type=float,
        default=0.3,
        help="Probability of horizontally flipping style images (style reference) (v0.5.3_ref)",
    )
    parser.add_argument(
        "--style_image_rotation_prob",
        type=float,
        default=0.3,
        help="Probability of rotating style images (style reference) (v0.5.3_ref)",
    )
    parser.add_argument(
        "--style_image_max_rotation_angle",
        type=float,
        default=15.0,
        help="Maximum rotation angle for style images (style reference) in degrees (v0.5.3_ref)",
    )
    parser.add_argument(
        "--style_latent_spatial_mask_prob",
        type=float,
        default=1,
        help="Probability of applying spatial masking to style latents (style reference) (v0.5.3_ref)",
    )
    parser.add_argument(
        "--style_latent_spatial_mask_max_ratio",
        type=float,
        default=0.3,
        help="Maximum ratio of spatial regions to mask in style latents (style reference) (v0.5.3_ref)",
    )
    parser.add_argument(
        "--style_latent_dropout_prob",
        type=float,
        default=0.1,
        help="Probability of completely dropping style latents (style reference) (v0.5.3_ref)",
    )
    parser.add_argument(
        "--style_latent_noise_scale",
        type=float,
        default=0.0,
        help="Scale of gaussian noise to add to style latents (style reference) (v0.5.3_ref)",
    )
    parser.add_argument(
        "--style_latent_channel_dropout_prob",
        type=float,
        default=0.0,
        help="Probability of dropping channels in style latents (style reference) (v0.5.3_ref)",
    )
    parser.add_argument(
        "--style_latent_channel_dropout_ratio",
        type=float,
        default=0.0,
        help="Ratio of channels to drop in style latents (style reference) (v0.5.3_ref)",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main(args=None):
    if args is None:
        args = parse_args()
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `hf auth login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
        # fsdp_plugin=fsd_plugin,
        # deepspeed_plugin=deepspeed_plugin,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id

    models = load_training_models(args, accelerator, logger)
    transformer = models.transformer
    vae = models.vae
    text_encoder = models.text_encoder
    tokenizer = models.tokenizer
    unwrap_model = models.unwrap_model

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    optimizer = create_optimizer(args, transformer, accelerator, logger)

    # Dataset and DataLoaders creation.
    style_image_aug_config = {
        'random_flip_prob': args.style_image_flip_prob,
        'random_rotation_prob': args.style_image_rotation_prob,
        'max_rotation_angle': args.style_image_max_rotation_angle,
    }
    logger.info(f"v0.5.4 Style image (style reference) augmentation config: {style_image_aug_config}")

    train_dataset = PanoWorldTrainingDataset(
        manifest=args.train_manifest,
        data_root=args.data_root,
        target_width=args.target_width,
        target_height=args.target_height,
        inputs_are_padded=not args.inputs_unpadded,
        style_image_aug_config=style_image_aug_config,
        curriculum_learning=args.curriculum_learning,
    )
    logger.info(f"v0.5.5 Curriculum learning: {'Enabled' if args.curriculum_learning else 'Disabled'}")
    if args.curriculum_learning:
        logger.info(f"  Stage 1 (Control only): 0 ~ {args.curriculum_stage1_steps} steps")
        logger.info(f"  Stage 2 (Mixed training): {args.curriculum_stage1_steps}+ steps")
        logger.info(f"    Style prob: {args.stage2_style_start_prob:.2f} → {args.stage2_style_end_prob:.2f} over {args.stage2_style_warmup_steps} steps")
        logger.info(f"    Mixing mode: {args.stage2_mixing_mode}")

    # Create data loaders.
    if not args.disable_bucket_training:
        # Use the bucketed dataset.
        bucket_dataset = BucketDataset(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True
        )
        train_dataloader = DataLoader(
            bucket_dataset,
            batch_size=args.train_batch_size,
            shuffle=False,  # The bucketed dataset handles shuffling internally.
            collate_fn=collate_training_batch,
            num_workers=args.dataloader_num_workers,
        )
        logger.info(f"Bucketed training dataset size: {len(bucket_dataset)}")
    else:
        # Use the original data loader.
            train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            collate_fn=collate_training_batch,
            num_workers=args.dataloader_num_workers,
        )

    # move back to cpu before deleting to ensure memory is freed see: https://github.com/huggingface/diffusers/issues/11376#issue-3008144624
    if args.cache_latents:
        vae = vae.to("cpu")
        del vae

    del text_encoder, tokenizer
    free_memory()
    torch.cuda.empty_cache()
    # Scheduler and math around the number of training steps.
    lr_scheduler, num_update_steps_per_epoch, overrode_max_train_steps = create_lr_scheduler(
        args, optimizer, accelerator, train_dataloader
    )

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )
    models.transformer = transformer

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "dreambooth-qwen-image-lora"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    # v0.5.3_ref: Configure latent augmentation for control and style images
    # Control latents (control image) - light augmentation
    control_latent_aug_config = {
        'dropout_prob': args.control_latent_dropout_prob,
        'spatial_mask_prob': args.control_latent_spatial_mask_prob,
        'spatial_mask_max_ratio': args.control_latent_spatial_mask_max_ratio,
        'noise_scale': args.control_latent_noise_scale,
    }

    # v0.5.3_ref: Style latents (style reference) - heavier augmentation to break spatial correspondence
    style_latent_aug_config = {
        'dropout_prob': args.style_latent_dropout_prob,
        'spatial_mask_prob': args.style_latent_spatial_mask_prob,
        'spatial_mask_max_ratio': args.style_latent_spatial_mask_max_ratio,
        'noise_scale': args.style_latent_noise_scale,
        'channel_dropout_prob': args.style_latent_channel_dropout_prob,
        'channel_dropout_ratio': args.style_latent_channel_dropout_ratio,
    }

    logger.info(f"v0.5.4 Control Latent augmentation config: {control_latent_aug_config}")
    logger.info(f"v0.5.4 Style Latent (style reference) augmentation config: {style_latent_aug_config}")
    logger.info(f"v0.5.4 Bidirectional training: DISABLED")

    # v0.5.4: Initialize curriculum learning state
    curriculum_stage = 1  # Start with Stage 1 (control only)
    stage_switched = False  # Track if we've switched to Stage 2

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()

        # Reshuffle the bucketed dataset at the beginning of each epoch.
        if not args.disable_bucket_training and hasattr(train_dataloader.dataset, 'reshuffle'):
            train_dataloader.dataset.reshuffle()
            logger.info(f"Epoch {epoch}: reshuffled bucketed dataset")

        for step, batch in enumerate(train_dataloader):
            if args.curriculum_learning and not stage_switched:
                if global_step >= args.curriculum_stage1_steps:   # Switch to Stage 2 after curriculum_stage1_steps.
                    curriculum_stage = 2
                    stage_switched = True

                    if hasattr(train_dataset, 'set_training_stage'):
                        train_dataset.set_training_stage(2)
                    elif hasattr(train_dataloader.dataset, 'base_dataset') and hasattr(train_dataloader.dataset.base_dataset, 'set_training_stage'):
                        train_dataloader.dataset.base_dataset.set_training_stage(2)

                    logger.info("=" * 80)
                    logger.info(f"🎓 CURRICULUM LEARNING STAGE SWITCH at step {global_step}")
                    logger.info(f"   Stage 1 (Control only) → Stage 2 (Mixed: Control + Style)")
                    logger.info(f"   Stage 2 style prob will increase: {args.stage2_style_start_prob:.2f} → {args.stage2_style_end_prob:.2f}")
                    logger.info("=" * 80)

            use_style_img = True
            if args.curriculum_learning and curriculum_stage == 1:
                use_style_img = False
                if step % args.curriculum_log_interval == 0:
                    logger.info(f"Stage 1 - Control only")
            elif args.curriculum_learning and curriculum_stage == 2:
                current_style_prob = compute_stage2_style_prob(
                    global_step=global_step,
                    stage1_steps=args.curriculum_stage1_steps,
                    start_prob=args.stage2_style_start_prob,
                    end_prob=args.stage2_style_end_prob,
                    warmup_steps=args.stage2_style_warmup_steps,
                    mode=args.stage2_mixing_mode
                )
                use_style_img = (random.random() < current_style_prob)

                if step % args.curriculum_log_interval == 0:
                    logger.info(f"Stage 2 - Style prob: {current_style_prob:.3f} ({current_style_prob*100:.1f}% use style) | This batch: {'With style' if use_style_img else 'Control only'}")

            # Get the current batch resolution.
            if not args.disable_bucket_training:
                bucket_resolution = batch["bucket_resolution"]
                if step % 100 == 0:
                    logger.info(f"Processing resolution: {bucket_resolution[0]}x{bucket_resolution[1]}")

            loss = train_one_batch(
                args=args,
                batch=batch,
                use_style_img=use_style_img,
                models=models,
                accelerator=accelerator,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                control_latent_aug_config=control_latent_aug_config,
                style_latent_aug_config=style_latent_aug_config,
            )

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        # Use a safer save path to avoid NCCL timeouts.
                        try:
                            # Save LoRA weights first.
                            transformer_unwrapped = unwrap_model(transformer)
                            transformer_lora_layers_to_save = get_peft_model_state_dict(transformer_unwrapped)


                            # Create the save directory.
                            os.makedirs(save_path, exist_ok=True)

                            # Save LoRA weights.
                            QwenImageEditPipeline.save_lora_weights(
                                save_path,
                                transformer_lora_layers=transformer_lora_layers_to_save,
                            )
                            logger.info(f"Successfully saved LoRA weights to {save_path}")
                            # Save training state such as optimizer and scheduler.
                            # Use a lightweight state checkpoint instead of saving the full model.
                            state_dict = {
                                "global_step": global_step,
                                "epoch": epoch,
                                "optimizer": optimizer.state_dict(),
                                "lr_scheduler": lr_scheduler.state_dict(),
                                "args": vars(args),
                            }
                            torch.save(state_dict, os.path.join(save_path, "training_state.pt"))

                            logger.info(f"Successfully saved checkpoint to {save_path}")

                        except Exception as e:
                            logger.error(f"Failed to save checkpoint: {e}")
                            # If custom saving fails, fall back to accelerator.save_state.
                            try:
                                accelerator.save_state(save_path)
                                logger.info(f"Saved with accelerator.save_state to {save_path}")
                            except Exception as e2:
                                logger.error(f"accelerator.save_state also failed: {e2}")
                                # Continue training even if checkpoint saving fails.
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Save the lora layers
    accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
