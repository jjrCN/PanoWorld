import importlib
import os
import time
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.distributed as dist
from rich import print
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from .lrm_train_setup import init_config, init_distributed, init_wandb_and_backup
from .training_utils import auto_resume_job, create_lr_scheduler, create_optimizer

try:
    import wandb
except ImportError:
    wandb = None


AMP_DTYPE_MAPPING = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "tf32": torch.float32,
}


@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for parameter in model.parameters():
        parameter.requires_grad = flag


def remove_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        key = key.replace("_checkpoint_wrapped_module.", "")
        key = key.replace("_orig_mod.", "")
        while key.startswith("module."):
            key = key[len("module.") :]
        new_state_dict[key] = value
    return new_state_dict


def load_symbol(dotted_path):
    module_name, symbol_name = dotted_path.rsplit(".", 1)
    return importlib.import_module(module_name).__dict__[symbol_name]


def build_dataloader(config):
    dataset_cls = load_symbol(config.training.get("dataset_name", "panoworld_lrm.lrm_train_dataset.Dataset"))
    dataset = dataset_cls(config)
    sampler = DistributedSampler(dataset)
    dataloader_kwargs = {
        "batch_size": config.training.batch_size_per_gpu,
        "shuffle": False,
        "num_workers": config.training.num_workers,
        "persistent_workers": config.training.num_workers > 0,
        "pin_memory": True,
        "drop_last": True,
        "sampler": sampler,
    }
    if config.training.num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = config.training.prefetch_factor
    dataloader = DataLoader(dataset, **dataloader_kwargs)
    return dataset, sampler, dataloader


def main():
    config = init_config()
    os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))

    ddp_info = init_distributed(seed=config.training.get("seed", 777))
    dist.barrier()

    if ddp_info.is_main_process:
        init_wandb_and_backup(config)
    dist.barrier()

    torch.backends.cuda.matmul.allow_tf32 = config.training.use_tf32
    torch.backends.cudnn.allow_tf32 = config.training.use_tf32

    dataset, data_sampler, dataloader = build_dataloader(config)
    if ddp_info.is_main_process:
        print("Total training scenes:", len(dataset.data_path))

    total_train_steps = config.training.train_steps
    grad_accum_steps = config.training.grad_accum_steps
    total_param_update_steps = total_train_steps
    total_train_steps = total_train_steps * grad_accum_steps
    model_cls = load_symbol(config.model.class_name)
    model = model_cls(config).to(ddp_info.device)
    ema = deepcopy(model).to(ddp_info.device)
    requires_grad(ema, False)
    model = DDP(model, device_ids=[ddp_info.local_rank])

    optimizer, optimized_param_dict, _ = create_optimizer(
        model,
        config.training.weight_decay,
        config.training.lr,
        (config.training.beta1, config.training.beta2),
    )
    optimized_params = list(optimized_param_dict.values())
    lr_scheduler = create_lr_scheduler(
        optimizer,
        total_param_update_steps,
        config.training.warmup,
        scheduler_type=config.training.get("scheduler_type", "cosine"),
    )

    ckpt_load_path = config.training.get("resume_ckpt", "") or config.training.checkpoint_dir
    optimizer, lr_scheduler, cur_train_step, cur_param_update_step = auto_resume_job(
        ckpt_load_path,
        model,
        optimizer,
        lr_scheduler,
        config.training.get("reset_training_state", False),
    )

    dist.barrier()
    update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()

    cur_epoch = cur_train_step // max(len(dataloader), 1)
    data_sampler.set_epoch(cur_epoch)
    dataloader_iter = iter(dataloader)
    start_train_step = cur_train_step
    use_wandb = bool(config.training.get("use_wandb", False)) and wandb is not None and ddp_info.is_main_process

    while cur_train_step <= total_train_steps:
        tic = time.time()
        try:
            data = next(dataloader_iter)
        except StopIteration:
            cur_epoch += 1
            if ddp_info.is_main_process:
                print(f"Epoch {cur_epoch}: rebuilding dataloader iterator")
            data_sampler.set_epoch(cur_epoch)
            dataloader_iter = iter(dataloader)
            data = next(dataloader_iter)

        batch = {k: v.to(ddp_info.device) for k, v in data.items() if isinstance(v, torch.Tensor)}
        input_data_dict = {key: value for key, value in batch.items() if "input" in key}
        target_data_dict = {key: value for key, value in batch.items() if "target" in key}
        input_room_ids = int(torch.unique(input_data_dict["input_room_ids"][0]).numel())
        input_view_num = len(input_data_dict["input_image_indexs"][0])
        target_view_num = len(target_data_dict["target_image_indexs"][0])

        with torch.autocast(
            enabled=config.training.use_amp,
            device_type="cuda",
            dtype=AMP_DTYPE_MAPPING[config.training.amp_dtype],
        ):
            ret_dict = model(input_data_dict, target_data_dict)

        update_grads = (cur_train_step + 1) % grad_accum_steps == 0 or cur_train_step == total_train_steps
        scaled_loss = ret_dict.loss_metrics.loss / grad_accum_steps
        if not update_grads:
            with model.no_sync():
                scaled_loss.backward()
        else:
            scaled_loss.backward()

        cur_train_step += 1
        skip_optimizer_step = bool(
            torch.isnan(ret_dict.loss_metrics.loss).item() or torch.isinf(ret_dict.loss_metrics.loss).item()
        )
        total_grad_norm = 0.0

        if update_grads:
            if not skip_optimizer_step and config.training.get("grad_clip_norm", 0) > 0:
                total_grad_norm = torch.nn.utils.clip_grad_norm_(
                    optimized_params,
                    max_norm=config.training.grad_clip_norm,
                ).item()
            if not skip_optimizer_step:
                optimizer.step()
                cur_param_update_step += 1
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()
            if not skip_optimizer_step:
                update_ema(ema, model.module)

        if not ddp_info.is_main_process:
            continue

        loss_dict = {k: float(f"{v.item():.6f}") for k, v in ret_dict.loss_metrics.items()}
        if (cur_train_step % config.training.print_every == 0) or (cur_train_step < 100 + start_train_step):
            message = (
                f"[Epoch {int(cur_epoch):>3d}] | Forward step: {int(cur_train_step):>6d} "
                f"(Param update step: {int(cur_param_update_step):>6d}) | "
                f"Iter Time: {time.time() - tic:.2f}s | LR: {optimizer.param_groups[0]['lr']:.6f} | "
                f"input_view_num: {input_view_num} | target_view_num: {target_view_num} | "
                f"input_room_ids: {input_room_ids}\n"
            )
            for key, value in loss_dict.items():
                message += f"{key}: {value:.6f} | "
            print(message)

        if use_wandb and (
            cur_train_step % config.training.wandb_log_every == 0
            or cur_train_step < 200 + start_train_step
        ):
            log_dict = {
                "iter": cur_train_step,
                "forward_pass_step": cur_train_step,
                "param_update_step": cur_param_update_step,
                "lr": optimizer.param_groups[0]["lr"],
                "iter_time": time.time() - tic,
                "grad_norm": total_grad_norm,
                "epoch": cur_epoch,
            }
            log_dict.update({"train/" + key: value for key, value in loss_dict.items()})
            wandb.log(log_dict, step=cur_train_step)

        if cur_train_step % config.training.checkpoint_every == 0 or cur_train_step == total_train_steps:
            checkpoint = {
                "model": remove_module_prefix(model.state_dict()),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "fwdbwd_pass_step": cur_train_step,
                "param_update_step": cur_param_update_step,
            }
            os.makedirs(config.training.checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(config.training.checkpoint_dir, f"ckpt_{cur_train_step:016}.pt")
            torch.save(checkpoint, ckpt_path)
            print(f"Saved checkpoint at step {cur_train_step} to {os.path.abspath(ckpt_path)}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
