"""Single-person BEDLAM dataset (bedlam_single), precomputed-latent based.

Each sample dir <seq>/<clip>/<frame>_b<inst>/ has:
  meta.npz     canon (N,3), uv (N,2, 512-crop coords), z (N,), cam_int, bbox, inst, caption
  caption.txt  Qwen2.5-VL caption (single-person)
  latent.npz   VAE posterior mean/std (run data/precompute_latents_run.sh first)

Returns per sample: latent_mean/std (VAE posterior), the mesh condition (canon/uv/z),
and the tokenized caption.
"""
from __future__ import annotations

import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SingleBedlamDataset(Dataset):
    def __init__(self, data_dir="data/bedlam_single", split="train", tokenizer=None,
                 proportion_empty_prompts=0.15):
        self.dir = data_dir
        self.tokenizer = tokenizer
        self.proportion_empty_prompts = proportion_empty_prompts
        man = json.load(open(os.path.join(data_dir, "manifest.json"), encoding="utf-8"))
        self.samples = man["samples"]
        self.indices = man["train_idx"] if split == "train" else man["test_idx"]

    def __len__(self):
        return len(self.indices)

    def _tokenize(self, caption):
        return self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt").input_ids[0]

    def __getitem__(self, i):
        d = os.path.join(self.dir, self.samples[self.indices[i]])

        z = np.load(os.path.join(d, "meta.npz"), allow_pickle=True)
        L = np.load(os.path.join(d, "latent.npz"))

        caption = open(os.path.join(d, "caption.txt"), encoding="utf-8").read().strip()
        if random.random() < self.proportion_empty_prompts:
            caption = ""

        return {
            "latent_mean": torch.from_numpy(L["mean"].astype(np.float32)),   # (4,64,64)
            "latent_std": torch.from_numpy(L["std"].astype(np.float32)),
            "canon": torch.from_numpy(z["canon"].astype(np.float32)),        # (N,3)
            "uv": torch.from_numpy(z["uv"].astype(np.float32)),              # (N,2)
            "z": torch.from_numpy(z["z"].astype(np.float32)),                # (N,)
            "input_ids": self._tokenize(caption),
        }


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def build_dataloader(cfg, tokenizer, split: str = "train") -> DataLoader:
    dataset = SingleBedlamDataset(
        data_dir=cfg.single_dir, split=split, tokenizer=tokenizer,
        proportion_empty_prompts=cfg.proportion_empty_prompts)
    return DataLoader(
        dataset, shuffle=(split == "train"), batch_size=cfg.train_batch_size,
        num_workers=cfg.num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=(split == "train"))
