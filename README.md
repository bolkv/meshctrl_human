# MeshCtrl-Human

Mesh-conditioned human image generation on a frozen Stable Diffusion 1.5 UNet.

An SMPL-X mesh is encoded per-vertex, splatted into a **camera-aligned frustum voxel grid**,
compressed to a 2D conditioning map, and injected through a ControlNet branch. The UNet, VAE
and text encoder stay frozen; only the mesh encoder + ControlNet are trained.

```
SMPL-X (10475 verts)
  canonical coords ──► FourierPE (27)  ─┐
  vertex normals   ──► FourierPE (27)  ─┴─► 54 ─► PointEncoder (1-ring attn) ─► 48
                                                                                │
                                      frustum trilinear splat (u,v,z)  ◄────────┘
                                                   │
                                  voxel (48, 128, 128, 32)
                                                   │  3D conv ↓, fold depth → channels
                                        cond map (320, 64, 64)
                                                   │
                        ControlNet trunk (t-aware) ─► 12 down + 1 mid residuals
                                                   │
                                   frozen SD1.5 UNet decoder
```

Conditioning the mesh branch with **per-vertex normals** (instead of a learned vertex-index
embedding) is what makes front/back facing recoverable: the facing cue survives the depth
fold because it lives in the feature *values*, not only in the depth *position*.

## Layout

```
train.py                   training (ControlNet + ShapeCtrl, UNet frozen; optional UNet LoRA)
sample_single.py           sample on the BEDLAM single-body test split  (GT | generated)
sample_traj100.py          sample on Trajectory100 in-the-wild mesh conditions
dataset_bedlam_single.py   dataloader (precomputed VAE latents + mesh tensors)

models/
  shapectrl.py             ShapeCtrl encoder + frustum_trilinear_scatter  ← core
  voxel_bridge.py          voxel → 2D ControlNet conditioning map
  point_transformer.py     PointEncoder over the fixed SMPL-X 1-ring
  controlnet.py            frozen base models, ControlNet build, bridge attach
  shapectrl_old.py         legacy pre-normal encoder (vertex_embed); only for `--old`

configs/model_config.py    all hyperparameters, overridable from the CLI

data/
  smplx_adjacency.pt       SMPL-X 1-ring neighbours (required)
  smplx_faces.npy          SMPL-X faces, for vertex normals (required)
```

Every dataclass field in `ModelConfig` becomes a `--flag`, e.g. `--train_batch_size 2`.

## Install

```bash
pip install -r requirements.txt
```

Base weights (`SG161222/Realistic_Vision_V5.1_noVAE`, ~470 MB) download from the Hub on first
run. Offline? copy an existing HF cache and point `HF_HOME` at it.

## Data layout (not shipped)

Training — `data/bedlam_single/`:

```
manifest.json                     {"samples": [...], "train_idx": [...], "test_idx": [...]}
<clip>/<seq>/<frame>_b<k>/
  meta.npz                        canon (N,3), uv (N,2) in 512-crop px, z (N,), cam_int, bbox
  caption.txt
  latent.npz                      VAE posterior mean/std (precomputed)
  img.jpg                         GT, for the sampling montages
```

Inference — `data/traj100/`:

```
manifest.json                     images / captions / ids ("<clip>#<frame>") / train_idx / test_idx
canon.npy  (T, 10475, 3)   uv.npy  (T, 10475, 2)   z.npy  (T, 10475)
```

## Train

```bash
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 \
accelerate launch --num_processes 2 --mixed_precision bf16 --main_process_port 29513 \
train.py --output_dir ./outputs/meshctrl_normal --use_lora false \
         --train_batch_size 2 --gradient_accumulation_steps 4
```

Checkpoints land in `<output_dir>/checkpoint-step-<N>/` every `--checkpointing_steps` (1500)
and hold both inference weights (`controlnet/`, `shapectrl.pt`) and full accelerate state.
Resume with `--resume_from_checkpoint <path>`.

**Multi-GPU on consumer cards (4090 and friends) needs `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1`.**
Without it the ShapeCtrl gradient all-reduce hangs until the watchdog aborts the run.

## Sample

```bash
# BEDLAM test split
python sample_single.py --ckpt outputs/meshctrl_normal/checkpoint-step-37500 \
       --n 6 --guidance 4.0 --cond 1.0 --out sample/bedlam.jpg

# Trajectory100, whole clip
python sample_traj100.py --ckpt outputs/meshctrl_normal/checkpoint-step-37500 \
       --clip 0013 --n 6 --cond 1.0 --out sample/clip13.jpg

# specific frames, or arbitrary frames across clips
python sample_traj100.py --clip 0013 --frames 40,44,48,50,52,56 ...
python sample_traj100.py --ids "0074#15,0086#95,0046#35" ...
```

### Timestep-scheduled conditioning (recommended)

Facing and global pose are decided early in denoising; texture quality is damaged by strong
conditioning late. A flat `--cond 1.5` fixes front/back but washes the image out. Splitting the
schedule gets both:

```bash
python sample_traj100.py --cond_hi 1.5 --cond_lo 0.4 --cond_t 500 ...
```

Strong (1.5) while `t >= 500`, weak (0.4) after. On a turning clip this tracked the ground-truth
facing on 5 of 6 frames with no loss of sharpness, where flat 0.7 got 2 of 6.

Other flags: `--no_text` (mesh-only; removes the front-facing bias that face/expression words in
the caption introduce), `--old` (load a legacy `vertex_embed` checkpoint).

## Known rough edges

- **Per-sample depth normalisation.** `frustum_trilinear_scatter` maps each sample's own
  `[z_min, z_max]` onto the 32 depth bins, so absolute distance is discarded and the number of
  bins separating a torso's front and back shrinks when a limb reaches toward the camera —
  exactly the poses where facing is hardest. A fixed depth range would decouple the two.
- **Sum instead of weighted average.** The scatter returns `acc_feat`, not `acc_feat / acc_w`,
  so a voxel's magnitude scales with how many vertices landed in it. That keeps density, but it
  also makes the conditioning strength depend on the subject's apparent size. `acc_w` is still
  accumulated and then unused — dead code worth either restoring or deleting.
- **fp16 accumulation.** The scatter accumulates in the feature dtype; under the fp16 sampling
  path a well-covered cell sums ~100 contributions. fp32 accumulation would be safer.
- **Crowd hallucination.** The conditioning says "a person is here" but never "nobody else is",
  so extra figures appear at the frame edges.
- **No temporal model.** Frames are generated independently, so identity drifts across a clip.
- `--config hands|coco` presets referenced in `model_config.py` are not part of this repo.
