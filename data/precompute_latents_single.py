"""Precompute VAE latent posteriors (mean/std) for bedlam_single -> per-sample latent.npz.

Lets training skip the VAE encode each step: train.py's `latent_mean` branch does
`latents = mean + std * noise` (scaling_factor applied there). Stored mean/std are the
RAW VAE posterior (pre-scaling), matching train.py's convention.

Sharded across GPUs (precompute_latents_run.sh). Resume: skip dirs that already have latent.npz.
Run AFTER captioning (needs GPU). Uses the training env (diffusers), no isolated libs.
"""
import os, sys, glob, numpy as np, cv2, torch
sys.path.insert(0, "/workspace/shapectrl")
from diffusers import AutoencoderKL
from configs.model_config import get_config

ROOT = os.environ.get("SINGLE_ROOT", "/workspace/shapectrl/data/bedlam_single")
BATCH = int(os.environ.get("BATCH", 32))
RES = 512


def load_vae(cfg):
    """Same VAE selection as models.controlnet.load_base_models."""
    if getattr(cfg, "vae_name_or_path", None):
        return AutoencoderKL.from_pretrained(cfg.vae_name_or_path)
    try:
        return AutoencoderKL.from_pretrained(
            cfg.pretrained_model_name_or_path, subfolder="vae", revision=cfg.revision)
    except Exception:
        return AutoencoderKL.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="vae")


def main():
    cfg = get_config([])
    vae = load_vae(cfg).to("cuda", torch.float16).eval()

    dirs = sorted(os.path.dirname(p) for p in glob.glob(f"{ROOT}/*/*/*_b*/img.jpg"))
    tasks = [d for d in dirs if not os.path.exists(f"{d}/latent.npz")]
    nshard = int(os.environ.get("NSHARD", "1")); shard = int(os.environ.get("SHARD", "0"))
    tasks = tasks[shard::nshard]
    print(f"[shard {shard}/{nshard}] {len(tasks)} imgs to encode", flush=True)

    ndone = nfail = 0
    for i in range(0, len(tasks), BATCH):
        batch = tasks[i:i + BATCH]
        try:
            imgs = []
            for d in batch:
                im = cv2.cvtColor(cv2.imread(f"{d}/img.jpg"), cv2.COLOR_BGR2RGB)
                if im.shape[0] != RES: im = cv2.resize(im, (RES, RES))
                imgs.append(im)
            px = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float() / 127.5 - 1.0
            px = px.to("cuda", torch.float16)
            with torch.no_grad():
                dist = vae.encode(px).latent_dist                      # DiagonalGaussian
                mean = dist.mean.cpu().numpy().astype(np.float16)      # (B,4,64,64) raw (pre-scaling)
                std = dist.std.cpu().numpy().astype(np.float16)
            for j, d in enumerate(batch):
                np.savez_compressed(f"{d}/latent.npz", mean=mean[j], std=std[j])
            ndone += len(batch)
        except Exception as e:
            nfail += len(batch); print(f"[FAIL] batch {i}: {e}", file=sys.stderr, flush=True)
        if (i // BATCH) % 20 == 0:
            print(f"[shard {shard}] {ndone}/{len(tasks)}", flush=True)
    print(f"[shard {shard}] done: {ndone} encoded, {nfail} failed", flush=True)


if __name__ == "__main__":
    main()
