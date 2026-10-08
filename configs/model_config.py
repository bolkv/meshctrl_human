from __future__ import annotations

import argparse
from dataclasses import dataclass, field, fields, MISSING
from typing import Optional


@dataclass
class ModelConfig:
    pretrained_model_name_or_path: str = "SG161222/Realistic_Vision_V5.1_noVAE"
    vae_name_or_path: Optional[str] = None 
    revision: Optional[str] = None
    controlnet_model_name_or_path: Optional[str] = None
    conditioning_channels: int = 3

    # Data
    data_dir: str = "data"
    single_dir: str = "data/bedlam_single"      # single-person dataset (meta.npz + caption.txt + latent.npz)
    midframe_dir: str = "data/bedlam_multi_train"
    adjacency_cache: str = "data/smplx_adjacency.pt"
    num_vertices: int = 10475
    max_bodies: int = 3
    drop_oob: bool = True
    resolution: int = 512
    proportion_empty_prompts: float = 0.15 
    mesh_dropout_prob: float = 0.15         
    num_workers: int = 8

    # Training
    output_dir: str = "./outputs/shapectrl_bedlam_single_lora"
    seed: int = 42
    train_batch_size: int = 4
    gradient_accumulation_steps: int = 2  
    num_train_epochs: int = 1000
    max_train_steps: Optional[int] = 50000 
    learning_rate: float = 5e-5
    lr_scheduler: str = "constant"         
    lr_warmup_steps: int = 500
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    set_grads_to_none: bool = True

    mixed_precision: str = "bf16"
    enable_xformers: bool = True
    allow_tf32: bool = True

    # appearance/shape disentangle experiment
    control_null_text: bool = True         # feed EMPTY text to ControlNet(ShapeCtrl) branch -> pure geometry
    use_lora: bool = True                  # train a UNet LoRA to absorb the (BEDLAM) appearance domain
    lora_rank: int = 16
    lora_lr: float = 1e-4                  # separate LR for the LoRA (usually higher than controlnet lr)

    
    logging_dir: str = "logs"
    report_to: str = "tensorboard"         
    tracker_project_name: str = "sd15-controlnet"
    checkpointing_steps: int = 1500          
    resume_from_checkpoint: Optional[str] = None  
    
    @classmethod
    def from_cli(cls, argv: Optional[list] = None) -> "ModelConfig":
        parser = argparse.ArgumentParser(description="SD1.5 ControlNet training config")
        for f in fields(cls):
            default = f.default if f.default is not MISSING else (
                f.default_factory() if f.default_factory is not MISSING else None  
            )
            if f.type == "bool" or isinstance(default, bool):
                parser.add_argument(f"--{f.name}", type=lambda x: x.lower() in ("1", "true", "yes"),
                                    default=default)
            elif isinstance(default, list):
                parser.add_argument(f"--{f.name}", nargs="*", default=default)
            elif default is None:
                parser.add_argument(f"--{f.name}", default=None)
            else:
                parser.add_argument(f"--{f.name}", type=type(default), default=default)
        args = parser.parse_args(argv)
        return cls(**vars(args))


def get_config(argv: Optional[list] = None) -> ModelConfig:
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    cls = ModelConfig
    if "--config" in argv:
        i = argv.index("--config")
        preset = argv[i + 1]
        del argv[i:i + 2]
        if preset == "hands":
            from configs.hands_config import HandsConfig
            cls = HandsConfig
        elif preset == "coco":
            from configs.coco_config import CocoConfig
            cls = CocoConfig
        elif preset not in ("base", "body"):
            raise ValueError(f"unknown --config preset: {preset}")
    return cls.from_cli(argv)
