"""SD1.5 + ControlNet model assembly.

Provides:
  - load_base_models(): frozen tokenizer / text_encoder / vae / unet + scheduler
  - build_controlnet(): trainable ControlNet (from UNet or from checkpoint)
  - build_inference_pipeline(): StableDiffusionControlNetPipeline for sampling

The ControlNet is the *only* trainable module; everything else is frozen,
matching the original ControlNet recipe (Zhang et al., 2023).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer

from configs.model_config import ModelConfig


@dataclass
class BaseModels:
    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModel
    vae: AutoencoderKL
    unet: UNet2DConditionModel
    noise_scheduler: DDPMScheduler


def load_base_models(cfg: ModelConfig) -> BaseModels:
    """Load the frozen SD1.5 backbone components."""
    tokenizer = CLIPTokenizer.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="tokenizer", revision=cfg.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="text_encoder", revision=cfg.revision
    )
    if getattr(cfg, "vae_name_or_path", None):
        vae = AutoencoderKL.from_pretrained(cfg.vae_name_or_path)   # explicit override (e.g. sd-vae-ft-mse)
    else:
        try:
            vae = AutoencoderKL.from_pretrained(
                cfg.pretrained_model_name_or_path, subfolder="vae", revision=cfg.revision
            )
        except Exception:
            # _noVAE base (e.g. RealisticVision) -> keep SD1.5 VAE so precomputed latents stay valid
            vae = AutoencoderKL.from_pretrained(
                "stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="vae"
            )
    unet = UNet2DConditionModel.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="unet", revision=cfg.revision
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="scheduler"
    )

    # Freeze everything; ControlNet is the only thing we train.
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    return BaseModels(tokenizer, text_encoder, vae, unet, noise_scheduler)


def build_controlnet(cfg: ModelConfig, unet: UNet2DConditionModel) -> ControlNetModel:
    """Create the trainable ControlNet.

    If cfg.controlnet_model_name_or_path is set, resume from that checkpoint;
    otherwise initialise a fresh ControlNet from the UNet weights (the standard
    way to bootstrap training — the zero-conv layers start at zero so the base
    model output is unchanged at step 0).
    """
    if cfg.controlnet_model_name_or_path:
        controlnet = ControlNetModel.from_pretrained(cfg.controlnet_model_name_or_path)
    else:
        controlnet = ControlNetModel.from_unet(
            unet, conditioning_channels=cfg.conditioning_channels
        )
    controlnet.train()
    return controlnet


def attach_voxel_bridge(
    controlnet: ControlNetModel, feat_dim: int = 24, depth: int = 32
) -> ControlNetModel:
    """Replace the default (image) conditioning embedding with the ShapeCtrl
    voxel bridge, so the 5-D voxel can be passed straight as `controlnet_cond`.
    """
    from models.voxel_bridge import VoxelControlEmbedding

    controlnet.controlnet_cond_embedding = VoxelControlEmbedding(
        in_channels=feat_dim,
        conditioning_embedding_channels=controlnet.config.block_out_channels[0],
        depth=depth,
    )
    return controlnet


def build_inference_pipeline(
    cfg: ModelConfig,
    controlnet: ControlNetModel,
    base: Optional[BaseModels] = None,
    torch_dtype: torch.dtype = torch.float16,
) -> StableDiffusionControlNetPipeline:
    """Assemble a sampling pipeline reusing already-loaded frozen modules.

    Used for in-training validation and standalone inference.
    """
    if base is not None:
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            cfg.pretrained_model_name_or_path,
            vae=base.vae,
            text_encoder=base.text_encoder,
            tokenizer=base.tokenizer,
            unet=base.unet,
            controlnet=controlnet,
            safety_checker=None,
            revision=cfg.revision,
            torch_dtype=torch_dtype,
        )
    else:
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            cfg.pretrained_model_name_or_path,
            controlnet=controlnet,
            safety_checker=None,
            revision=cfg.revision,
            torch_dtype=torch_dtype,
        )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe
