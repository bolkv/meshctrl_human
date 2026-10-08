#!/usr/bin/env python3
"""Build multi-person BEDLAM_multi training set (ragged, per-frame variable #bodies).

For each frame (512 letterbox jpg): every visible body's camera-frame SMPL-X verts, plus
per-body global_orient (for canonicalisation) and the frame's cam intrinsics. The dataloader
derives canon/uv/z from these on the fly (cheap). Bodies are depth-sorted -> inst slot.

Outputs data/bedlam_multi_train/:
  verts.npy       (T,10475,3) fp16   camera-frame vertices, all bodies concatenated
  body_go.npy     (T,3)       fp32   global_orient per body (canon = verts @ R(go) - mean)
  body_frame.npy  (T,)        int32  frame index per body
  body_inst.npy   (T,)        uint8  instance slot (0..K-1, depth order front->back)
  frame_K.npy     (F,3,3)     fp32   cam intrinsics per frame
  latent_mean.npy (F,4,64,64) fp16   VAE latent mean of the 512 image
  latent_std.npy  (F,4,64,64) fp16
  manifest.json   {images, captions, train_idx, test_idx}   (frame-level; 50K train / rest test)
"""
import glob, json, os, sys
import cv2, numpy as np, torch, smplx
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL
import project_bedlam as pb

OUT = "bedlam_multi_train"; ROOT = "BEDLAM_multi"
RES = 512; N_TRAIN = 50000; SMPL_CHUNK = 256; VAE_CHUNK = 32
DEV = "cuda"
LIMIT = int(os.environ.get("LIMIT", 0))
os.makedirs(OUT, exist_ok=True)

_gm = {g: smplx.create(pb.MODELS, "smplx", gender=g, ext="npz", num_betas=10,
        flat_hand_mean=True, use_pca=False, batch_size=1).to(DEV).eval() for g in ["male", "female"]}

@torch.no_grad()
def verts_batch(pose, beta, trans, gender):
    m = _gm[gender]; t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=DEV)
    o = m(betas=t(beta), global_orient=t(pose[:, :3]), body_pose=t(pose[:, 3:66]),
          left_hand_pose=t(pose[:, 75:120]), right_hand_pose=t(pose[:, 120:165]),
          jaw_pose=t(pose[:, 66:69]), leye_pose=t(pose[:, 69:72]), reye_pose=t(pose[:, 72:75]),
          expression=torch.zeros(len(pose), 10, device=DEV), transl=t(trans))
    return o.vertices.cpu().numpy()

clips = sorted(glob.glob(f"{ROOT}/*/seq_*"))
if LIMIT: clips = clips[:LIMIT]
frames, bodies = [], []
clip_of_frame = []
for cdir in clips:
    try:
        d = np.load(f"{cdir}/meta.npz", allow_pickle=True)
        cap = json.load(open(f"{cdir}/captions.json", encoding="utf-8")) if os.path.exists(f"{cdir}/captions.json") else {}
    except Exception:
        continue
    imgs = [str(x) for x in d["images"]]; genders = [str(g) for g in d["genders"]]
    for fi in range(len(imgs)):
        vis = np.where(d["visible"][:, fi])[0]
        if len(vis) == 0:
            continue
        frame_id = len(frames)
        for b in vis:
            bodies.append(dict(frame=frame_id, pose=d["pose_cam"][b, fi].astype(np.float32),
                               beta=d["betas"][b].astype(np.float32),
                               trans=d["trans_cam"][b, fi].astype(np.float32), gender=genders[b]))
        frames.append(dict(img=f"{cdir}/{imgs[fi]}", cap=cap.get(imgs[fi], ""),
                           K=d["cam_int"][fi].astype(np.float32)))
        clip_of_frame.append(cdir)
F, T = len(frames), len(bodies)
print(f"pass1: {len(clips)} clips -> {F} frames, {T} bodies", flush=True)

verts = np.empty((T, 10475, 3), np.float16)
body_go = np.stack([b["pose"][:3] for b in bodies]).astype(np.float32)
body_frame = np.array([b["frame"] for b in bodies], np.int32)
depth = np.empty(T, np.float32)
for g in ["male", "female"]:
    idx = [i for i, b in enumerate(bodies) if b["gender"] == g]
    for s in range(0, len(idx), SMPL_CHUNK):
        chunk = idx[s:s + SMPL_CHUNK]
        V = verts_batch(np.stack([bodies[i]["pose"] for i in chunk]),
                        np.stack([bodies[i]["beta"] for i in chunk]),
                        np.stack([bodies[i]["trans"] for i in chunk]), g)
        for j, i in enumerate(chunk):
            verts[i] = V[j].astype(np.float16); depth[i] = V[j][:, 2].mean()
    print(f"  smplx {g} done", flush=True)

body_inst = np.zeros(T, np.uint8)
from collections import defaultdict
byframe = defaultdict(list)
for i in range(T): byframe[body_frame[i]].append(i)
for fid, ids in byframe.items():
    for slot, i in enumerate(sorted(ids, key=lambda k: depth[k])):
        body_inst[i] = slot

tf = transforms.Compose([transforms.Resize(RES), transforms.CenterCrop(RES),
                         transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
vae = AutoencoderKL.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5",
                                    subfolder="vae").to(DEV, torch.float16).eval()
lat_m = np.empty((F, 4, RES // 8, RES // 8), np.float16); lat_s = np.empty_like(lat_m)
for s in range(0, F, VAE_CHUNK):
    e = min(s + VAE_CHUNK, F)
    x = torch.stack([tf(Image.open(frames[j]["img"]).convert("RGB")) for j in range(s, e)]).to(DEV, torch.float16)
    with torch.no_grad():
        dist = vae.encode(x).latent_dist
    lat_m[s:e] = dist.mean.cpu().numpy().astype(np.float16)
    lat_s[s:e] = dist.std.cpu().numpy().astype(np.float16)
    if s % (VAE_CHUNK * 50) == 0: print(f"  vae {s}/{F}", flush=True)

uniq = sorted(set(clip_of_frame))
perm = np.random.default_rng(42).permutation(len(uniq))
cum, train_clips = 0, set()
counts = {c: clip_of_frame.count(c) for c in uniq}
for k in perm:
    c = uniq[k]
    if cum < N_TRAIN:
        train_clips.add(c); cum += counts[c]
train_idx = [i for i in range(F) if clip_of_frame[i] in train_clips]
test_idx = [i for i in range(F) if clip_of_frame[i] not in train_clips]

np.save(f"{OUT}/verts.npy", verts); np.save(f"{OUT}/body_go.npy", body_go)
np.save(f"{OUT}/body_frame.npy", body_frame); np.save(f"{OUT}/body_inst.npy", body_inst)
np.save(f"{OUT}/frame_K.npy", np.stack([f["K"] for f in frames]))
np.save(f"{OUT}/latent_mean.npy", lat_m); np.save(f"{OUT}/latent_std.npy", lat_s)
json.dump({"images": [f["img"] for f in frames], "captions": [f["cap"] for f in frames],
           "train_idx": train_idx, "test_idx": test_idx},
          open(f"{OUT}/manifest.json", "w"), ensure_ascii=False)
print(f"saved {OUT}/  frames={F} bodies={T}  train={len(train_idx)} test={len(test_idx)}", flush=True)
