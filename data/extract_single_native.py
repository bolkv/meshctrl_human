"""Single-person crops from NATIVE BEDLAM mp4 frames (no resize, no quality loss).

Reuses precomputed bedlam_multi_train verts (camera-space, resolution-independent),
recovers the NATIVE camera K from the stored 512-letterbox K, projects to native
pixels, then crops a person-centered square straight out of the native mp4 frame
with black padding and NO resize.

Per body (in a 1~3-person frame): drop if padding >= 20% (native bounds) or too small.

Structured output (OUT):
  <seq>/<clip>/<frame>_b<inst>/
      img.jpg      native person crop (square, black-padded, NOT resized)
      meta.npz     canon(N,3), uv(N,2 native-crop coords), z(N), cam_int(3,3), bbox, inst, caption
  manifest.json
"""
import os, sys, re, json, glob, subprocess, shutil, numpy as np, cv2, torch
import project_bedlam as pb

SRC = "bedlam_multi_train"
OUT = os.environ.get("OUT", "/workspace/shapectrl/data/bedlam_single")
WORK = os.environ.get("WORK", "/tmp/claude-0/-workspace/3673b842-a4de-4e64-a083-84288609462f/scratchpad/native_work")
STRIDE = 5; MARGIN = 1.3; MAXPAD = 0.20; MINPX = 40; TARGET = 512
LIMIT_CLIPS = int(os.environ.get("LIMIT_CLIPS", "0"))
dev = "cuda"

m = json.load(open(f"{SRC}/manifest.json")); imgs = m["images"]; caps = m["captions"]
bf = np.load(f"{SRC}/body_frame.npy")
inst_all = np.load(f"{SRC}/body_inst.npy")
K512 = np.load(f"{SRC}/frame_K.npy").astype(np.float32)
go_all = np.load(f"{SRC}/body_go.npy").astype(np.float32)
verts = np.load(f"{SRC}/verts.npy", mmap_mode="r")
B = verts.shape[0]

def seqof(p):
    for part in p.split("/"):
        if re.match(r"^2022\d{4}_", part): return part
def parts_of(p):
    a = p.split("/"); return seqof(p), a[-2], os.path.splitext(a[-1])[0]

def native_WH(seq): return (720, 1280) if "closeup" in seq else (1280, 720)
Knat = np.zeros_like(K512); WHn = np.zeros((len(imgs), 2), np.float32)
for f, p in enumerate(imgs):
    W, H = native_WH(seqof(p)); WHn[f] = (W, H)
    s = 512 / max(W, H); nw, nh = round(W * s), round(H * s)
    px, py = (512 - nw) // 2, (512 - nh) // 2
    Kc = K512[f]; Kn = Kc.copy()
    Kn[0, 0] = Kc[0, 0] / s; Kn[1, 1] = Kc[1, 1] / s
    Kn[0, 2] = (Kc[0, 2] - px) / s; Kn[1, 2] = (Kc[1, 2] - py) / s
    Knat[f] = Kn

Kb = torch.tensor(Knat[bf], device=dev); WHb = torch.tensor(WHn[bf], device=dev)
keep = np.zeros(B, bool); crop = np.zeros((B, 3), np.int32)
bs = 4000
for s0 in range(0, B, bs):
    e = min(s0 + bs, B)
    v = torch.tensor(np.asarray(verts[s0:e], np.float32), device=dev)
    z = v[..., 2]
    uv = (Kb[s0:e] @ (v / z.unsqueeze(-1)).transpose(1, 2)).transpose(1, 2)[..., :2]
    fin = torch.isfinite(uv).all(2).all(1); zbad = (z <= 0.1).any(1)
    x0 = uv[..., 0].amin(1); x1 = uv[..., 0].amax(1); y0 = uv[..., 1].amin(1); y1 = uv[..., 1].amax(1)
    w = x1 - x0; h = y1 - y0
    sz = torch.clamp(torch.maximum(w, h) * MARGIN, min=1)
    bx0 = (x0 + x1) / 2 - sz / 2; by0 = (y0 + y1) / 2 - sz / 2
    W = WHb[s0:e, 0]; H = WHb[s0:e, 1]
    ix = torch.clamp(torch.minimum(bx0 + sz, W) - torch.clamp(bx0, min=0), min=0)
    iy = torch.clamp(torch.minimum(by0 + sz, H) - torch.clamp(by0, min=0), min=0)
    pf = 1 - (ix * iy) / (sz * sz)
    ok = fin & (~zbad) & (h > MINPX) & (pf < MAXPAD) & (sz < 6000)
    keep[s0:e] = ok.cpu().numpy()
    crop[s0:e, 0] = bx0.round().cpu().numpy(); crop[s0:e, 1] = by0.round().cpu().numpy()
    crop[s0:e, 2] = sz.round().cpu().numpy()

keep_idx = np.nonzero(keep)[0]
print(f"kept {len(keep_idx):,} / {B:,} bodies ({keep.sum()/B*100:.1f}%) [native bounds]", flush=True)

PER_PERSON = int(os.environ.get("PER_PERSON", "2"))
from collections import defaultdict
by_person = defaultdict(list)
for b in keep_idx:
    seq, clip, frame = parts_of(imgs[int(bf[b])])
    by_person[(seq, clip, int(inst_all[b]))].append((int(frame), b))
sel = []
for k, lst in by_person.items():
    lst.sort()
    if len(lst) <= PER_PERSON:
        sel += [b for _, b in lst]
    else:
        pick = np.linspace(0, len(lst) - 1, PER_PERSON).round().astype(int)
        sel += [lst[i][1] for i in sorted(set(pick))]
keep_idx = np.array(sorted(sel))
print(f"-> {len(by_person):,} persons, {PER_PERSON}/person => {len(keep_idx):,} samples", flush=True)

by_clip = defaultdict(list)
for b in keep_idx:
    seq, clip, frame = parts_of(imgs[int(bf[b])])
    by_clip[(seq, clip)].append(b)
clips = sorted(by_clip)
if LIMIT_CLIPS: clips = clips[:LIMIT_CLIPS]
print(f"{len(clips):,} clips to process", flush=True)

def rodrigues(go): return cv2.Rodrigues(go.astype(np.float64))[0].astype(np.float32)

def crop_native(img, bx0, by0, sz):
    H, W = img.shape[:2]
    pad = max(0, -bx0, -by0, bx0 + sz - W, by0 + sz - H)
    if pad > 0:
        imgp = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return imgp[by0 + pad:by0 + pad + sz, bx0 + pad:bx0 + pad + sz]
    return img[by0:by0 + sz, bx0:bx0 + sz]

os.makedirs(WORK, exist_ok=True)
sample_dirs = []; frame_of = []
cur_seqdir = None; seq_wd = None
n_done = 0
for (seq, clip) in clips:
    if seq != cur_seqdir:
        if seq_wd: shutil.rmtree(seq_wd, ignore_errors=True)
        cur_seqdir = seq; seq_wd = f"{WORK}/{seq}"; os.makedirs(seq_wd, exist_ok=True)
        subprocess.run(['tar', 'xf', f"{pb.ROOT}/{seq}/mp4/{seq}_mp4.tar", '-C', seq_wd], check=False)
    mp4 = f"{seq_wd}/{seq}/mp4/{clip}.mp4"
    if not os.path.exists(mp4):
        continue
    fwd = f"{seq_wd}/_f_{clip}"; os.makedirs(fwd, exist_ok=True)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', mp4, '-vf', f"select='not(mod(n,{STRIDE}))'",
                    '-vsync', '0', '-start_number', '0', f'{fwd}/%04d.png'], check=False)
    fcache = {}
    for b in by_clip[(seq, clip)]:
        f = int(bf[b]); _, _, frame = parts_of(imgs[f]); fi = int(frame)
        inst = int(inst_all[b]); bx0, by0, sz = (int(x) for x in crop[b])
        if fi not in fcache:
            fcache[fi] = cv2.imread(f"{fwd}/{fi:04d}.png")
        img = fcache[fi]
        if img is None: continue
        cr = crop_native(img, bx0, by0, sz)
        interp = cv2.INTER_AREA if sz > TARGET else cv2.INTER_CUBIC
        cr = cv2.resize(cr, (TARGET, TARGET), interpolation=interp)
        s = TARGET / sz
        rel = f"{seq}/{clip}/{frame}_b{inst}"; d = f"{OUT}/{rel}"; os.makedirs(d, exist_ok=True)
        cv2.imwrite(f"{d}/img.jpg", cr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        v = np.asarray(verts[b], np.float32); K = Knat[f]
        uvf = (K @ (v / v[:, 2:3]).T).T[:, :2]
        uvc = (uvf - np.array([bx0, by0], np.float32)) * s
        c = v @ rodrigues(go_all[b]); canon = c - c.mean(0, keepdims=True)
        Kc = K.copy(); Kc[0, 0] *= s; Kc[1, 1] *= s
        Kc[0, 2] = (K[0, 2] - bx0) * s; Kc[1, 2] = (K[1, 2] - by0) * s
        np.savez_compressed(f"{d}/meta.npz",
            canon=canon.astype(np.float16), uv=uvc.astype(np.float16), z=v[:, 2].astype(np.float16),
            cam_int=Kc.astype(np.float32), bbox=np.array([bx0, by0, sz], np.int32),
            inst=inst, caption=caps[f])
        sample_dirs.append(rel); frame_of.append(f)
    shutil.rmtree(fwd, ignore_errors=True)
    n_done += 1
    if n_done % 200 == 0: print(f"  {n_done:,}/{len(clips):,} clips, {len(sample_dirs):,} samples", flush=True)
if seq_wd: shutil.rmtree(seq_wd, ignore_errors=True)

train_frames = set(m["train_idx"]); fo = np.array(frame_of)
train_mask = np.array([int(x) in train_frames for x in fo]) if len(fo) else np.array([], bool)
json.dump({
    "samples": sample_dirs, "captions": [caps[int(x)] for x in fo],
    "train_idx": np.nonzero(train_mask)[0].tolist(), "test_idx": np.nonzero(~train_mask)[0].tolist(),
}, open(f"{OUT}/manifest.json", "w"))
print(f"DONE: {len(sample_dirs):,} native samples -> {OUT}", flush=True)
