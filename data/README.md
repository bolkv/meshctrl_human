# BEDLAM preprocessing

Builds `data/bedlam_single/`, the dataset `dataset_bedlam_single.py` trains on.

Raw BEDLAM gives per-sequence SMPL-X motion plus camera tracks. The pipeline poses the
meshes, projects them into each frame, cuts one square crop per visible person straight out
of the native mp4 (no resize), then attaches a caption and a precomputed VAE latent to every
crop.

```
BEDLAM (mp4 + gendered_ground_truth) + SMPL-X body models
        │  download_bedlam_labels.sh, download_smplx.sh
        ▼
  project_bedlam.py            library: be_seq.csv + motion_seq.npz + camera csv
        │                      -> posed SMPL-X verts in camera frame, 30fps
        ▼
  build_bedlam_multi.py        data/bedlam_multi_train/
        │                      verts.npy, body_go.npy, body_frame.npy, body_inst.npy,
        │                      frame_K.npy, latent_mean.npy, manifest.json (+ train/test split)
        ▼
  extract_single_native.py     data/bedlam_single/<seq>/<clip>/<frame>_b<inst>/
        │                        img.jpg    native person crop, square, black-padded
        │                        meta.npz   canon (N,3), uv (N,2), z (N,), cam_int, bbox, inst
        │                      + manifest.json   (train_idx / test_idx inherited from upstream)
        ▼
  caption_single_qwen.py       caption.txt   (Qwen2.5-VL-7B)
        ▼
  precompute_latents_single.py latent.npz    (VAE posterior mean/std, pre-scaling)
```

Each stage skips work that is already on disk, so all of them are resumable.

## Run order

```bash
# 0. raw assets
./download_smplx.sh                      # SMPL-X body models -> data/models/smplx
./download_bedlam_labels.sh              # BEDLAM labels; the mp4s are downloaded separately

# 1. pose + project + pack (GPU: SMPL-X forward and VAE encode)
python build_bedlam_multi.py             # LIMIT=<n> to smoke-test on a few sequences

# 2. native single-person crops
OUT=$PWD/bedlam_single python extract_single_native.py
#    PER_PERSON=2   crops kept per person per clip
#    LIMIT_CLIPS=5  smoke test
#    WORK=...       scratch dir for extracted frames (set this, see Paths below)

# 3. captions (sharded over GPUs)
NGPU=4 ./caption_single_run.sh

# 4. VAE latents (sharded over GPUs)
NGPU=4 ./precompute_latents_run.sh
```

Steps 3 and 4 shard with `NSHARD` / `SHARD`; the `*_run.sh` wrappers set those from `NGPU`.

## Paths to change on a new machine

These scripts were written against `/workspace/shapectrl` and still carry that default.
Most read an environment variable first, so exporting these is enough:

| Variable | Used by | Default |
|---|---|---|
| `BEDLAM_ROOT` | `project_bedlam.py` | `/workspace/shapectrl/data/BEDLAM` |
| `SMPLX_MODELS` | `project_bedlam.py` | `/workspace/shapectrl/data/models` |
| `OUT` | `extract_single_native.py` | `/workspace/shapectrl/data/bedlam_single` |
| `WORK` | `extract_single_native.py` | a scratch dir that will not exist here |
| `SINGLE_ROOT` | captioning, latents | `/workspace/shapectrl/data/bedlam_single` |
| `QWEN_MODEL` | `caption_single_qwen.py` | `.../data/models/Qwen2.5-VL-7B-Instruct` |

One path is hardcoded and must be edited by hand:

- `precompute_latents_single.py` line 11 — `sys.path.insert(0, "/workspace/shapectrl")`,
  which is how it reaches `configs.model_config`. Point it at this repo root.

`extract_single_native.py` does `import project_bedlam`, so run it from this directory.

## Extra dependencies

Beyond the repo's `requirements.txt`:

- `smplx`, `trimesh`, `torchvision`, `Pillow` — mesh posing and image IO
- captioning needs **transformers 4.49** and `qwen_vl_utils`, which conflicts with the
  training environment's 4.46. The original setup kept them apart with
  `PYTHONPATH=<repo>/data/.qwen_libs`; `caption_single_run.sh` expects that layout. Install
  them into a separate directory (or a separate venv) and adjust the wrapper.

## Not included

`build_bedlam_multi_cc.py` (center-crop variant of stage 1) and the multi-person loader are
left out: this repo trains on single-person crops only.
