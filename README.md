# MeshCtrl-Human

Mesh-conditioned human image generation on a frozen Stable Diffusion 1.5 UNet.

An SMPL-X mesh is encoded per-vertex, splatted into a **camera-aligned frustum voxel grid**,
compressed to a 2D conditioning map, and injected through a ControlNet branch. The UNet, VAE
and text encoder stay frozen; only the mesh encoder and the ControlNet are trained.

```
SMPL-X (10475 verts)
  vertex index ──► Embedding (21) ─┐
  canonical xyz ─► FourierPE (27) ─┴─► 48 ─► PointEncoder (1-ring attn) ─► 48
                                                                           │
                                 frustum trilinear splat (u,v,z)  ◄────────┘
                                              │   weighted average per cell
                             voxel (48, 128, 128, 32)
                                              │  3D conv ↓, fold depth → channels
                                   cond map (320, 64, 64)
                                              │
                   ControlNet trunk (t-aware) ─► 12 down + 1 mid residuals
                                              │
                              frozen SD1.5 UNet decoder
```

The grid is frustum-aligned rather than object-aligned so that its H and W axes are the image
axes: `H=W=128` halves once inside the bridge and lands exactly on the 64x64 latent, so nothing
has to learn the projection. The depth axis is folded into channels by the bridge.

## Layout

```
train.py                   training (ControlNet + ShapeCtrl, UNet frozen; optional UNet LoRA)
sample_single.py           sample on the BEDLAM single-body test split  (GT | generated)
dataset_bedlam_single.py   dataloader (precomputed VAE latents + mesh tensors)

models/
  shapectrl.py             ShapeCtrl encoder + frustum_trilinear_scatter   <- core
  voxel_bridge.py          voxel -> 2D ControlNet conditioning map
  point_transformer.py     PointEncoder over the fixed SMPL-X 1-ring
  controlnet.py            frozen base models, ControlNet build, bridge attach

configs/model_config.py    all hyperparameters, overridable from the CLI

data/
  smplx_adjacency.pt       SMPL-X 1-ring neighbours (required)
  smplx_faces.npy          SMPL-X faces (optional; only to rebuild the adjacency cache)
  README.md                BEDLAM preprocessing pipeline
  *.py, *.sh               the preprocessing stages themselves
```

Every dataclass field in `ModelConfig` becomes a `--flag`, e.g. `--train_batch_size 2`.

## Install

```bash
pip install -r requirements.txt
```

Base weights (`SG161222/Realistic_Vision_V5.1_noVAE`, ~470 MB) download from the Hub on first
run. Offline? copy an existing HF cache and point `HF_HOME` at it.

## Data

The dataset itself is not shipped. `data/` holds the BEDLAM preprocessing pipeline that
builds it - see **[data/README.md](data/README.md)** for the stage order, the environment
variables to repoint, and the one hardcoded path that needs editing.

Layout it produces, `data/bedlam_single/`:

```
manifest.json                     {"samples": [...], "train_idx": [...], "test_idx": [...]}
<clip>/<seq>/<frame>_b<k>/
  meta.npz                        canon (N,3), uv (N,2) in 512-crop px, z (N,), cam_int, bbox
  caption.txt
  latent.npz                      VAE posterior mean/std (precomputed)
  img.jpg                         GT, for the sampling montages
```

`canon` is the mesh de-rotated by its own global orientation; `uv` and `z` are the camera
projection of the posed mesh and carry all of the view information.

## Train

```bash
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 \
accelerate launch --num_processes 2 --mixed_precision bf16 --main_process_port 29513 \
train.py --output_dir ./outputs/meshctrl --use_lora false \
         --train_batch_size 2 --gradient_accumulation_steps 4
```

Checkpoints land in `<output_dir>/checkpoint-step-<N>/` every `--checkpointing_steps` (1500)
and hold both inference weights (`controlnet/`, `shapectrl.pt`) and full accelerate state.
Resume with `--resume_from_checkpoint <path>`.

**Multi-GPU on consumer cards (4090 and friends) needs `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1`.**
Without it the ShapeCtrl gradient all-reduce hangs until the NCCL watchdog aborts the run.

## Sample

```bash
python sample_single.py --ckpt outputs/meshctrl/checkpoint-step-30000 \
       --n 6 --guidance 4.0 --cond 1.0 --out sample/bedlam.jpg
```

`--cond` is the ControlNet conditioning scale and is the main quality/control knob. Low values
let the SD prior override the mesh on anything it finds ambiguous (most visibly, which way the
person faces); high values enforce the mesh but wash out texture, because global structure is
settled early in denoising while texture is formed late. If a single value will not give you
both, scale it by timestep rather than raising it flat.

## Known rough edges

- **Per-sample depth normalisation.** `frustum_trilinear_scatter` maps each sample's own
  `[z_min, z_max]` onto the 32 depth bins, so absolute distance is discarded and the number of
  bins separating a torso's front and back shrinks when a limb reaches toward the camera -
  exactly the poses where facing is hardest. A fixed depth range would decouple the two.
- **fp16 accumulation.** The scatter accumulates in the feature dtype; under the fp16 sampling
  path a well-covered cell sums ~100 contributions. fp32 accumulation would be safer.
- **No surface orientation.** The conditioning encodes where each vertex is but not which way
  its surface points, so front/back is only recoverable from the depth ordering of the two
  surfaces - a weak cue that the SD front-facing prior often wins against.
- **Crowd hallucination.** The conditioning says "a person is here" but never "nobody else is",
  so extra figures appear at the frame edges.
- **No temporal model.** Frames are generated independently, so identity drifts across a clip.
- `--config hands|coco` presets referenced in `model_config.py` are not part of this repo.
