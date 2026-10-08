from __future__ import annotations

import logging
import math
import os
import shutil

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import is_xformers_available
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from safetensors.torch import save_file
from tqdm.auto import tqdm

from configs.model_config import ModelConfig, get_config
from dataset_bedlam_single import build_dataloader
from models.controlnet import attach_voxel_bridge, build_controlnet, load_base_models
from models.shapectrl import ShapeCtrl

logger = get_logger(__name__)


def _weight_dtype(accelerator: Accelerator) -> torch.dtype:
    if accelerator.mixed_precision == "fp16":
        return torch.float16
    if accelerator.mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def main():
    cfg: ModelConfig = get_config()

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with=cfg.report_to,
        project_config=ProjectConfiguration(
            project_dir=cfg.output_dir,
            logging_dir=os.path.join(cfg.output_dir, cfg.logging_dir),
        ),
        # LoRA leaves most of the UNet frozen -> some params get no grad under DDP
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=cfg.use_lora)],
    )
    logging.basicConfig(level=logging.INFO)
    logger.info(accelerator.state, main_process_only=False)
    if cfg.seed is not None:
        set_seed(cfg.seed)
    if accelerator.is_main_process:
        os.makedirs(cfg.output_dir, exist_ok=True)
    if cfg.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Models
    base = load_base_models(cfg)                          # frozen vae/unet/text + scheduler
    controlnet = build_controlnet(cfg, base.unet)
    shapectrl = ShapeCtrl(
        num_vertices=cfg.num_vertices,
        adjacency_cache=cfg.adjacency_cache,
        drop_oob=cfg.drop_oob,
    )
    attach_voxel_bridge(controlnet, feat_dim=shapectrl.out_channels, depth=shapectrl.D)

    if cfg.enable_xformers and is_xformers_available():
        base.unet.enable_xformers_memory_efficient_attention()
        controlnet.enable_xformers_memory_efficient_attention()
    if cfg.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()

    weight_dtype = _weight_dtype(accelerator)
    base.vae.to(accelerator.device, dtype=weight_dtype)
    base.unet.to(accelerator.device, dtype=weight_dtype)
    base.text_encoder.to(accelerator.device, dtype=weight_dtype)

    # UNet LoRA to absorb the (BEDLAM) appearance domain; base UNet stays frozen.
    lora_params = []
    if cfg.use_lora:
        base.unet.add_adapter(LoraConfig(
            r=cfg.lora_rank, lora_alpha=cfg.lora_rank, init_lora_weights="gaussian",
            target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
        cast_training_params(base.unet, dtype=torch.float32)     # LoRA params -> fp32
        if cfg.gradient_checkpointing:
            base.unet.enable_gradient_checkpointing()
        lora_params = [p for p in base.unet.parameters() if p.requires_grad]

    # empty-text embedding for the ControlNet(ShapeCtrl) branch -> pure geometry
    null_embed = None
    if cfg.control_null_text:
        with torch.no_grad():
            null_ids = base.tokenizer(
                "", max_length=base.tokenizer.model_max_length, padding="max_length",
                truncation=True, return_tensors="pt").input_ids.to(accelerator.device)
            null_embed = base.text_encoder(null_ids, return_dict=False)[0]   # (1,77,C)

    # Optimizer / data / scheduler  (separate LR for the LoRA)
    params = list(controlnet.parameters()) + list(shapectrl.parameters()) + lora_params
    param_groups = [{"params": list(controlnet.parameters()) + list(shapectrl.parameters()),
                     "lr": cfg.learning_rate}]
    if lora_params:
        param_groups.append({"params": lora_params, "lr": cfg.lora_lr})
    optimizer = torch.optim.AdamW(
        param_groups, betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay, eps=cfg.adam_epsilon,
    )

    train_dataloader = build_dataloader(cfg, base.tokenizer, split="train")

    # optimizer steps per epoch accounting for DDP world size (so multi-GPU
    # doesn't inflate the epoch count). len() here is the un-sharded dataloader.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / (cfg.gradient_accumulation_steps * accelerator.num_processes))
    if cfg.max_train_steps is None:
        max_train_steps = cfg.num_train_epochs * num_update_steps_per_epoch
        num_train_epochs = cfg.num_train_epochs
    else:
        max_train_steps = cfg.max_train_steps
        num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        cfg.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )

    if cfg.use_lora:
        controlnet, shapectrl, base.unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            controlnet, shapectrl, base.unet, optimizer, train_dataloader, lr_scheduler)
    else:
        controlnet, shapectrl, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            controlnet, shapectrl, optimizer, train_dataloader, lr_scheduler)

    if accelerator.is_main_process:
        accelerator.init_trackers(cfg.tracker_project_name, config=vars(cfg))

    logger.info(f"***** Training: {len(train_dataloader.dataset)} samples, "
                f"{num_train_epochs} epochs, {max_train_steps} steps *****")


    global_step = 0
    first_epoch = 0
    resume_step = 0
    if cfg.resume_from_checkpoint:
        path = cfg.resume_from_checkpoint
        if path == "latest":
            dirs = sorted([d for d in os.listdir(cfg.output_dir) if d.startswith("checkpoint")],
                          key=lambda x: int(x.split("-")[-1]))
            path = os.path.join(cfg.output_dir, dirs[-1]) if dirs else None
        if path:
            accelerator.print(f"resuming from {path}")
            accelerator.load_state(path)                      # models + optimizer + scheduler + RNG
            global_step = int(os.path.basename(path).split("-")[-1])
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = global_step % num_update_steps_per_epoch  # update steps already done this epoch

    accum_models = [controlnet, shapectrl] + ([base.unet] if cfg.use_lora else [])
    for epoch in range(first_epoch, num_train_epochs):
        controlnet.train(); shapectrl.train()
        if cfg.use_lora:
            base.unet.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch,
                            disable=not accelerator.is_local_main_process,
                            desc=f"epoch {epoch + 1}/{num_train_epochs}")
        # on the resumed epoch, skip the batches already trained
        if epoch == first_epoch and resume_step > 0:
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader, resume_step * cfg.gradient_accumulation_steps)
            progress_bar.update(resume_step)
        else:
            active_dataloader = train_dataloader
        for batch in active_dataloader:
            with accelerator.accumulate(*accum_models):
                mean = batch["latent_mean"].to(dtype=weight_dtype)
                std = batch["latent_std"].to(dtype=weight_dtype)
                latents = (mean + std * torch.randn_like(std)) * base.vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, base.noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=latents.device).long()
                noisy_latents = base.noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = base.text_encoder(batch["input_ids"], return_dict=False)[0]
                # ControlNet(ShapeCtrl) sees EMPTY text (pure geometry); UNet keeps the caption
                ctrl_hidden = (null_embed.to(weight_dtype).expand(bsz, -1, -1)
                               if cfg.control_null_text else encoder_hidden_states)

                # ShapeCtrl: single-person mesh -> frustum voxel (B,24,H,W,D)
                voxel = shapectrl(batch["canon"], batch["uv"], batch["z"])

                # mesh condition dropout (p_m): zero the voxel on a fraction of samples
                # so the model learns eps_phi(.,0); enables g_mesh = eps(.,m) - eps(.,0) at inference.
                if cfg.mesh_dropout_prob > 0:
                    keep = (torch.rand(voxel.shape[0], device=voxel.device) >= cfg.mesh_dropout_prob)
                    voxel = voxel * keep.view(-1, *([1] * (voxel.dim() - 1)))

                # ControlNet residuals (bridge consumes the voxel as controlnet_cond)
                down_res, mid_res = controlnet(
                    noisy_latents, timesteps,
                    encoder_hidden_states=ctrl_hidden,
                    controlnet_cond=voxel.to(weight_dtype),
                    return_dict=False,
                )

                model_pred = base.unet(
                    noisy_latents, timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[r.to(weight_dtype) for r in down_res],
                    mid_block_additional_residual=mid_res.to(weight_dtype),
                    return_dict=False,
                )[0]

                if base.noise_scheduler.config.prediction_type == "v_prediction":
                    target = base.noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params, cfg.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=cfg.set_grads_to_none)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step % cfg.checkpointing_steps == 0:
                    _save(cfg, accelerator, controlnet, shapectrl, global_step, unet=base.unet, tag="step")

            if global_step >= max_train_steps:
                break

    accelerator.wait_for_everyone()
    _save(cfg, accelerator, controlnet, shapectrl, global_step, unet=base.unet, final=True)
    accelerator.end_training()


def _save(cfg, accelerator, controlnet, shapectrl, num, unet=None, final=False, tag="step"):
    """Save a checkpoint. Must be called on ALL processes (save_state is collective).

    Writes both a resumable accelerate state (models + optimizer + scheduler + RNG)
    and a clean inference export (controlnet/ + shapectrl.pt + unet_lora.safetensors).
    """
    name = "final" if final else f"checkpoint-{tag}-{num}"
    out = os.path.join(cfg.output_dir, name)
    if not final:
        accelerator.save_state(out)                      # collective: resumable state
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.makedirs(out, exist_ok=True)
        accelerator.unwrap_model(controlnet).save_pretrained(os.path.join(out, "controlnet"))
        torch.save(accelerator.unwrap_model(shapectrl).state_dict(),
                   os.path.join(out, "shapectrl.pt"))
        if cfg.use_lora and unet is not None:            # droppable appearance-domain LoRA
            lora_sd = get_peft_model_state_dict(accelerator.unwrap_model(unet))
            save_file(lora_sd, os.path.join(out, "unet_lora.safetensors"))
        logger.info(f"saved {out}")


if __name__ == "__main__":
    main()
