"""Model-side helpers for PanoWorld LoRA training."""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from accelerate import DistributedType
from peft import LoraConfig, prepare_model_for_kbit_training, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor

from diffusers import (
    AutoencoderKLQwenImage,
    BitsAndBytesConfig,
    FlowMatchEulerDiscreteScheduler,
    QwenImageEditPlusPipeline,
    QwenImagePipeline,
    QwenImageTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _collate_lora_metadata,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    offload_models,
)
from diffusers.utils import convert_unet_state_dict_to_peft
from diffusers.utils.torch_utils import is_compiled_module


@dataclass
class TrainingModels:
    tokenizer: Any
    noise_scheduler: Any
    noise_scheduler_copy: Any
    vae: Any
    vae_scale_factor: int
    latents_mean: torch.Tensor
    latents_std: torch.Tensor
    text_encoder: Any
    processor: Any
    transformer: Any
    text_encoding_pipeline: Any
    transformer_lora_config: LoraConfig
    weight_dtype: torch.dtype
    unwrap_model: Any


def resolve_weight_dtype(mixed_precision: str) -> torch.dtype:
    """Resolve the non-trainable model dtype used by Accelerate mixed precision."""

    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def load_training_models(args, accelerator, logger) -> TrainingModels:
    """Load Qwen Image Edit components and attach the transformer LoRA adapter."""

    tokenizer = Qwen2Tokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    weight_dtype = resolve_weight_dtype(accelerator.mixed_precision)

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        revision=args.revision,
        shift=3.0,
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    vae = AutoencoderKLQwenImage.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    vae_scale_factor = 2 ** len(vae.temperal_downsample)
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(accelerator.device)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
        accelerator.device
    )
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    processor = Qwen2VLProcessor.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="processor",
        revision=args.revision,
    )

    quantization_config = None
    if args.bnb_quantization_config_path is not None:
        with open(args.bnb_quantization_config_path, "r", encoding="utf-8") as handle:
            config_kwargs = json.load(handle)
        if config_kwargs.get("load_in_4bit"):
            config_kwargs["bnb_4bit_compute_dtype"] = weight_dtype
        quantization_config = BitsAndBytesConfig(**config_kwargs)

    transformer = QwenImageTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
        quantization_config=quantization_config,
        torch_dtype=weight_dtype,
    )
    if args.bnb_quantization_config_path is not None:
        transformer = prepare_model_for_kbit_training(transformer, use_gradient_checkpointing=False)

    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    to_kwargs = {"dtype": weight_dtype, "device": accelerator.device} if not args.offload else {"dtype": weight_dtype}
    vae.to(**to_kwargs)
    text_encoder.to(**to_kwargs)
    transformer_to_kwargs = (
        {"device": accelerator.device}
        if args.bnb_quantization_config_path is not None
        else {"device": accelerator.device, "dtype": weight_dtype}
    )
    if accelerator.distributed_type != DistributedType.FSDP:
        transformer.to(**transformer_to_kwargs)

    text_encoding_pipeline = QwenImageEditPlusPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=None,
        transformer=None,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=None,
        processor=processor,
    )

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    target_modules = [layer.strip() for layer in args.lora_layers.split(",")] if args.lora_layers else [
        "to_k",
        "to_q",
        "to_v",
        "to_out.0",
    ]
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(transformer_lora_config)

    unwrap_model = register_lora_state_hooks(
        accelerator=accelerator,
        transformer=transformer,
        transformer_lora_config=transformer_lora_config,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        mixed_precision=args.mixed_precision,
        logger=logger,
    )

    if args.mixed_precision == "fp16":
        cast_training_params([transformer], dtype=torch.float32)

    return TrainingModels(
        tokenizer=tokenizer,
        noise_scheduler=noise_scheduler,
        noise_scheduler_copy=noise_scheduler_copy,
        vae=vae,
        vae_scale_factor=vae_scale_factor,
        latents_mean=latents_mean,
        latents_std=latents_std,
        text_encoder=text_encoder,
        processor=processor,
        transformer=transformer,
        text_encoding_pipeline=text_encoding_pipeline,
        transformer_lora_config=transformer_lora_config,
        weight_dtype=weight_dtype,
        unwrap_model=unwrap_model,
    )


def create_optimizer(args, transformer, accelerator, logger):
    """Create the optimizer for trainable LoRA parameters."""

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    params_to_optimize = [{"params": [param for param in transformer.parameters() if param.requires_grad], "lr": args.learning_rate}]
    if args.optimizer.lower() not in {"prodigy", "adamw"}:
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and args.optimizer.lower() != "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError as exc:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                ) from exc
            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        return optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    try:
        import prodigyopt
    except ImportError as exc:
        raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`") from exc

    if args.learning_rate <= 0.1:
        logger.warning("Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0")

    return prodigyopt.Prodigy(
        params_to_optimize,
        betas=(args.adam_beta1, args.adam_beta2),
        beta3=args.prodigy_beta3,
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
        decouple=args.prodigy_decouple,
        use_bias_correction=args.prodigy_use_bias_correction,
        safeguard_warmup=args.prodigy_safeguard_warmup,
    )


def create_lr_scheduler(args, optimizer, accelerator, train_dataloader):
    """Create the LR scheduler and update max_train_steps when it is epoch-derived."""

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    overrode_max_train_steps = False
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    return scheduler, num_update_steps_per_epoch, overrode_max_train_steps


def apply_latent_augmentation(latents: torch.Tensor, config: dict[str, float]) -> torch.Tensor:
    """Apply optional dropout, masking, and noise to VAE latents."""

    if not config or not any(config.values()):
        return latents

    device = latents.device
    dtype = latents.dtype
    batch, channels, frames, height, width = latents.shape

    dropout_prob = float(config.get("dropout_prob", 0.0))
    if dropout_prob > 0 and random.random() < dropout_prob:
        return torch.zeros_like(latents)

    spatial_mask_prob = float(config.get("spatial_mask_prob", 0.0))
    spatial_mask_max_ratio = float(config.get("spatial_mask_max_ratio", 0.3))
    if spatial_mask_prob > 0 and random.random() < spatial_mask_prob:
        mask = torch.ones((batch, 1, frames, height, width), device=device, dtype=dtype)
        for item_index in range(batch):
            for _ in range(random.randint(1, 3)):
                actual_mask_ratio = random.uniform(0.1, spatial_mask_max_ratio)
                block_h = int(height * actual_mask_ratio)
                block_w = int(width * actual_mask_ratio)
                start_h = random.randint(0, max(0, height - block_h))
                start_w = random.randint(0, max(0, width - block_w))
                mask[item_index, :, :, start_h : start_h + block_h, start_w : start_w + block_w] = 0
        latents = latents * mask

    channel_dropout_prob = float(config.get("channel_dropout_prob", 0.0))
    channel_dropout_ratio = float(config.get("channel_dropout_ratio", 0.2))
    if channel_dropout_prob > 0 and random.random() < channel_dropout_prob:
        num_drop_channels = int(channels * channel_dropout_ratio)
        if num_drop_channels > 0:
            drop_indices = random.sample(range(channels), num_drop_channels)
            mask = torch.ones((batch, channels, 1, 1, 1), device=device, dtype=dtype)
            mask[:, drop_indices] = 0
            latents = latents * mask

    noise_scale = float(config.get("noise_scale", 0.0))
    if noise_scale > 0:
        latents = latents + torch.randn_like(latents) * noise_scale

    return latents


def compute_stage2_style_prob(
    global_step: int,
    stage1_steps: int,
    start_prob: float,
    end_prob: float,
    warmup_steps: int,
    mode: str = "linear",
) -> float:
    """Return the current style-reference probability for curriculum stage 2."""

    if global_step < stage1_steps:
        return 0.0
    steps_in_stage2 = global_step - stage1_steps
    if steps_in_stage2 >= warmup_steps:
        return end_prob

    progress = steps_in_stage2 / warmup_steps
    if mode == "cosine":
        return start_prob + (end_prob - start_prob) * (1 - math.cos(progress * math.pi)) / 2
    if mode == "exp":
        return start_prob + (end_prob - start_prob) * (progress**2)
    return start_prob + (end_prob - start_prob) * progress


def compute_text_embeddings(
    prompts: list[str],
    text_encoding_pipeline: Any,
    images_per_prompt: list[list[Any]],
    max_sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode prompt plus conditioning images and pad the batch to one sequence length."""

    with torch.no_grad():
        all_prompt_embeds = []
        all_prompt_embeds_masks = []

        for prompt, images in zip(prompts, images_per_prompt):
            if len(images) == 2:
                system_prompt = (
                    "Generate a new image based on the spatial structure and furniture layout of the first image "
                    "(control) and the partial reconstruction results of the second image (view). Follow the "
                    "following description"
                )
            else:
                system_prompt = (
                    "Generate a new image based on the spatial structure and furniture layout of the first image "
                    "(control) and the partial reconstruction results of the second image (view). Follow the style, "
                    "shape, material, color, texture of the third image (style reference). Follow the following "
                    "description"
                )

            prompt_embeds, prompt_embeds_mask = text_encoding_pipeline.encode_prompt(
                prompt=system_prompt + ": " + prompt,
                max_sequence_length=max_sequence_length,
                image=images,
            )
            all_prompt_embeds.append(prompt_embeds.squeeze(0))
            all_prompt_embeds_masks.append(prompt_embeds_mask.squeeze(0))

        max_seq_len = max(emb.shape[0] for emb in all_prompt_embeds)
        padded_embeds = []
        padded_masks = []
        for emb, mask in zip(all_prompt_embeds, all_prompt_embeds_masks):
            seq_len = emb.shape[0]
            if seq_len < max_seq_len:
                padded_embeds.append(F.pad(emb, (0, 0, 0, max_seq_len - seq_len), value=0))
                padded_masks.append(F.pad(mask, (0, max_seq_len - seq_len), value=0))
            else:
                padded_embeds.append(emb)
                padded_masks.append(mask)

    return torch.stack(padded_embeds, dim=0), torch.stack(padded_masks, dim=0)


def get_sigmas(noise_scheduler, timesteps: torch.Tensor, device: torch.device, n_dim: int = 4, dtype=torch.float32):
    """Select scheduler sigmas and reshape them for latent arithmetic."""

    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == timestep).nonzero().item() for timestep in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def train_one_batch(
    args,
    batch: dict[str, Any],
    use_style_img: bool,
    models: TrainingModels,
    accelerator,
    optimizer,
    lr_scheduler,
    control_latent_aug_config: dict[str, float],
    style_latent_aug_config: dict[str, float],
) -> torch.Tensor:
    """Run one optimization step while preserving the released condition order."""

    transformer = models.transformer
    vae = models.vae
    noise_scheduler = models.noise_scheduler_copy
    vae_scale_factor = models.vae_scale_factor
    latents_mean = models.latents_mean
    latents_std = models.latents_std
    weight_dtype = models.weight_dtype

    with accelerator.accumulate([transformer]):
        with offload_models(models.text_encoding_pipeline, device=accelerator.device, offload=args.offload):
            if use_style_img:
                images_per_prompt = [
                    [memory, proxy, style]
                    for memory, proxy, style in zip(
                        batch["visual_memory_cond_pil"],
                        batch["geometric_proxy_cond_pil"],
                        batch["style_img_cond_pil"],
                    )
                ]
            else:
                images_per_prompt = [
                    [memory, proxy]
                    for memory, proxy in zip(batch["visual_memory_cond_pil"], batch["geometric_proxy_cond_pil"])
                ]
            prompt_embeds, prompt_embeds_mask = compute_text_embeddings(
                batch["captions"],
                models.text_encoding_pipeline,
                images_per_prompt,
                args.max_sequence_length,
            )

        with offload_models(vae, device=accelerator.device, offload=args.offload):
            pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
            model_input = vae.encode(pixel_values).latent_dist.sample()

            visual_memory_vae = batch["visual_memory_vae"].to(dtype=vae.dtype)
            visual_memory_latents = vae.encode(visual_memory_vae).latent_dist.sample()

            geometric_proxy_vae = batch["geometric_proxy_vae"].to(dtype=vae.dtype)
            geometric_proxy_latents = vae.encode(geometric_proxy_vae).latent_dist.sample()

            if use_style_img:
                style_img_vae = batch["style_img_vae"].to(dtype=vae.dtype)
                style_latents = vae.encode(style_img_vae).latent_dist.sample()
                del pixel_values, visual_memory_vae, geometric_proxy_vae, style_img_vae
            else:
                style_latents = None
                del pixel_values, visual_memory_vae, geometric_proxy_vae

        model_input = ((model_input - latents_mean) * latents_std).to(dtype=weight_dtype)

        if not args.cache_latents:
            visual_memory_latents = ((visual_memory_latents - latents_mean) * latents_std).to(dtype=weight_dtype)
            geometric_proxy_latents = ((geometric_proxy_latents - latents_mean) * latents_std).to(dtype=weight_dtype)
            if use_style_img and style_latents is not None:
                style_latents = ((style_latents - latents_mean) * latents_std).to(dtype=weight_dtype)

        visual_memory_latents = apply_latent_augmentation(visual_memory_latents, control_latent_aug_config)
        geometric_proxy_latents = apply_latent_augmentation(geometric_proxy_latents, control_latent_aug_config)
        if use_style_img and style_latents is not None:
            style_latents = apply_latent_augmentation(style_latents, style_latent_aug_config)

        noise = torch.randn_like(model_input)
        batch_size = model_input.shape[0]
        timestep_density = compute_density_for_timestep_sampling(
            weighting_scheme=args.weighting_scheme,
            batch_size=batch_size,
            logit_mean=args.logit_mean,
            logit_std=args.logit_std,
            mode_scale=args.mode_scale,
        )
        indices = (timestep_density * noise_scheduler.config.num_train_timesteps).long()
        timesteps = noise_scheduler.timesteps[indices].to(device=model_input.device)

        sigmas = get_sigmas(noise_scheduler, timesteps, accelerator.device, n_dim=model_input.ndim, dtype=model_input.dtype)
        noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

        bucket_resolution = batch["bucket_resolution"]
        visual_memory_vae_size = batch["visual_memory_vae_size"]
        geometric_proxy_vae_size = batch["geometric_proxy_vae_size"]
        if use_style_img:
            style_vae_size = batch["style_vae_size"]
            img_shapes = [
                [
                    (1, bucket_resolution[1] // vae_scale_factor // 2, bucket_resolution[0] // vae_scale_factor // 2),
                    (
                        1,
                        visual_memory_vae_size[1] // vae_scale_factor // 2,
                        visual_memory_vae_size[0] // vae_scale_factor // 2,
                    ),
                    (
                        1,
                        geometric_proxy_vae_size[1] // vae_scale_factor // 2,
                        geometric_proxy_vae_size[0] // vae_scale_factor // 2,
                    ),
                    (1, style_vae_size[1] // vae_scale_factor // 2, style_vae_size[0] // vae_scale_factor // 2),
                ]
            ] * batch_size
        else:
            img_shapes = [
                [
                    (1, bucket_resolution[1] // vae_scale_factor // 2, bucket_resolution[0] // vae_scale_factor // 2),
                    (
                        1,
                        visual_memory_vae_size[1] // vae_scale_factor // 2,
                        visual_memory_vae_size[0] // vae_scale_factor // 2,
                    ),
                    (
                        1,
                        geometric_proxy_vae_size[1] // vae_scale_factor // 2,
                        geometric_proxy_vae_size[0] // vae_scale_factor // 2,
                    ),
                ]
            ] * batch_size

        noisy_model_input = noisy_model_input.permute(0, 2, 1, 3, 4)
        packed_noisy_model_input = QwenImageEditPlusPipeline._pack_latents(
            noisy_model_input,
            batch_size=model_input.shape[0],
            num_channels_latents=noisy_model_input.shape[2],
            height=noisy_model_input.shape[3],
            width=noisy_model_input.shape[4],
        )

        visual_memory_latents = visual_memory_latents.permute(0, 2, 1, 3, 4)
        packed_visual_memory_latents = QwenImageEditPlusPipeline._pack_latents(
            visual_memory_latents,
            batch_size=model_input.shape[0],
            num_channels_latents=visual_memory_latents.shape[2],
            height=visual_memory_latents.shape[3],
            width=visual_memory_latents.shape[4],
        )

        geometric_proxy_latents = geometric_proxy_latents.permute(0, 2, 1, 3, 4)
        packed_geometric_proxy_latents = QwenImageEditPlusPipeline._pack_latents(
            geometric_proxy_latents,
            batch_size=model_input.shape[0],
            num_channels_latents=geometric_proxy_latents.shape[2],
            height=geometric_proxy_latents.shape[3],
            width=geometric_proxy_latents.shape[4],
        )

        if use_style_img and style_latents is not None:
            style_latents = style_latents.permute(0, 2, 1, 3, 4)
            packed_style_latents = QwenImageEditPlusPipeline._pack_latents(
                style_latents,
                batch_size=model_input.shape[0],
                num_channels_latents=style_latents.shape[2],
                height=style_latents.shape[3],
                width=style_latents.shape[4],
            )
            packed_input = torch.cat(
                [
                    packed_noisy_model_input,
                    packed_visual_memory_latents,
                    packed_geometric_proxy_latents,
                    packed_style_latents,
                ],
                dim=1,
            )
        else:
            packed_input = torch.cat(
                [packed_noisy_model_input, packed_visual_memory_latents, packed_geometric_proxy_latents], dim=1
            )

        model_pred = transformer(
            hidden_states=packed_input,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_embeds_mask,
            timestep=timesteps / 1000,
            img_shapes=img_shapes,
            txt_seq_lens=prompt_embeds_mask.sum(dim=1).tolist(),
            return_dict=False,
        )[0]
        model_pred = model_pred[:, : packed_noisy_model_input.size(1)]
        model_pred = QwenImageEditPlusPipeline._unpack_latents(
            model_pred, bucket_resolution[1], bucket_resolution[0], vae_scale_factor
        )

        weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
        target = noise - model_input
        if args.with_prior_preservation:
            model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
            target, target_prior = torch.chunk(target, 2, dim=0)
            prior_loss = torch.mean(
                (weighting.float() * (model_pred_prior.float() - target_prior.float()) ** 2).reshape(
                    target_prior.shape[0], -1
                ),
                1,
            ).mean()

        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        ).mean()
        if args.with_prior_preservation:
            loss = loss + args.prior_loss_weight * prior_loss

        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    return loss


def unwrap_accelerated_model(accelerator, model):
    """Return the original module behind Accelerate and optional torch.compile wrappers."""

    model = accelerator.unwrap_model(model)
    return model._orig_mod if is_compiled_module(model) else model


def register_lora_state_hooks(
    accelerator,
    transformer,
    transformer_lora_config,
    pretrained_model_name_or_path: str,
    mixed_precision: str,
    logger,
):
    """Register Accelerate hooks that save and load only transformer LoRA weights."""

    def unwrap_model(model):
        return unwrap_accelerated_model(accelerator, model)

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            modules_to_save = {}

            for model in models:
                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    model = unwrap_model(model)
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                    modules_to_save["transformer"] = model
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                if weights:
                    weights.pop()

            QwenImagePipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
                **_collate_lora_metadata(modules_to_save),
            )

    def load_model_hook(models, input_dir):
        transformer_for_load = None

        if accelerator.distributed_type != DistributedType.DEEPSPEED:
            while len(models) > 0:
                model = models.pop()
                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    transformer_for_load = unwrap_model(model)
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")
        else:
            transformer_for_load = QwenImageTransformer2DModel.from_pretrained(
                pretrained_model_name_or_path, subfolder="transformer"
            )
            transformer_for_load.add_adapter(transformer_lora_config)

        lora_state_dict = QwenImagePipeline.lora_state_dict(input_dir)
        transformer_state_dict = {
            key.replace("transformer.", ""): value
            for key, value in lora_state_dict.items()
            if key.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(
            transformer_for_load, transformer_state_dict, adapter_name="default"
        )
        unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None) if incompatible_keys is not None else None
        if unexpected_keys:
            logger.warning(
                "Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                f"{unexpected_keys}."
            )

        if mixed_precision == "fp16":
            cast_training_params([transformer_for_load])

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    return unwrap_model
