"""Sample the single-body meshctrl_bedlam_normal model on a few Trajectory100 frames.
Left = mesh uv projection (pose ref, red=near/green=far) | right = generated."""
import os, json, argparse, numpy as np, torch, cv2
from safetensors.torch import load_file
from diffusers import UniPCMultistepScheduler
from configs.model_config import get_config
from models.controlnet import load_base_models, build_controlnet, attach_voxel_bridge
from models.shapectrl import ShapeCtrl

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="outputs/meshctrl_bedlam_normal/checkpoint-step-4500")
ap.add_argument("--n", type=int, default=6)
ap.add_argument("--guidance", type=float, default=4.0)
ap.add_argument("--cond", type=float, default=0.7)
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="sample/traj100_normal_4500.jpg")
ap.add_argument("--clip", default=None, help="e.g. 0013 -> only frames from that clip")
ap.add_argument("--frames", default=None, help="comma frame numbers within clip, e.g. 44,48,52")
ap.add_argument("--no_text", action="store_true", help="empty prompt: mesh-only conditioning")
ap.add_argument("--old", action="store_true", help="use old vertex_embed (no-normal) ShapeCtrl")
ap.add_argument("--cond_hi", type=float, default=None, help="t-scheduled cond: scale for t>=cond_t")
ap.add_argument("--cond_lo", type=float, default=None, help="t-scheduled cond: scale for t<cond_t")
ap.add_argument("--cond_t", type=int, default=500, help="timestep threshold for cond schedule")
ap.add_argument("--ids", default=None, help="explicit clip#frame ids across clips, comma-sep")
a = ap.parse_args()
dev, dt = "cuda", torch.float16
cfg = get_config([])

base = load_base_models(cfg)
if a.old:
    from models.shapectrl_old import ShapeCtrlOld
    sc = ShapeCtrlOld(num_vertices=cfg.num_vertices, adjacency_cache=cfg.adjacency_cache, drop_oob=cfg.drop_oob)
else:
    sc = ShapeCtrl(num_vertices=cfg.num_vertices, adjacency_cache=cfg.adjacency_cache, drop_oob=cfg.drop_oob)
cn = build_controlnet(cfg, base.unet)
attach_voxel_bridge(cn, feat_dim=sc.out_channels, depth=sc.D)
cn.load_state_dict(load_file(f"{a.ckpt}/controlnet/diffusion_pytorch_model.safetensors"))
sc.load_state_dict(torch.load(f"{a.ckpt}/shapectrl.pt", map_location="cpu"))
for m in (base.vae, base.unet, base.text_encoder, cn): m.to(dev, dt).eval()
sc.to(dev, dt).eval()
sched = UniPCMultistepScheduler.from_config(base.noise_scheduler.config)
tok = base.tokenizer

man = json.load(open("data/traj100/manifest.json"))
canon = np.load("data/traj100/canon.npy", mmap_mode="r")
uv = np.load("data/traj100/uv.npy", mmap_mode="r")
z = np.load("data/traj100/z.npy", mmap_mode="r")
if a.ids is not None:
    idset = a.ids.split(",")
    id2i = {man["ids"][i]: i for i in range(len(man["ids"]))}
    pick = [id2i[x] for x in idset if x in id2i]
elif a.clip is not None:
    idxs = [i for i in range(len(man["ids"])) if str(man["ids"][i]).split("#")[0] == a.clip]
    print(f"clip {a.clip}: {len(idxs)} frames")
    if a.frames is not None:                                       # explicit frame numbers within clip
        want = set(a.frames.split(","))
        pick = [i for i in idxs if str(man["ids"][i]).split("#")[1] in want]
    else:
        pick = [idxs[k] for k in np.linspace(0, len(idxs) - 1, a.n).astype(int)]
else:
    idxs = man["test_idx"]
    pick = [idxs[k] for k in np.linspace(0, len(idxs) - 1, a.n).astype(int)]


def emb(t):
    i = tok(t, max_length=77, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(dev)
    return base.text_encoder(i, return_dict=False)[0]


def mesh_ref(u, zz):
    im = np.full((512, 512, 3), 255, np.uint8)
    zn = (zz - zz.min()) / (zz.max() - zz.min() + 1e-6)
    for k in range(0, len(u), 8):
        x, y = int(u[k, 0]), int(u[k, 1])
        if 0 <= x < 512 and 0 <= y < 512:
            c = int(255 * (1 - zn[k])); cv2.circle(im, (x, y), 1, (0, c, 255 - c), -1)
    return im


@torch.no_grad()
def gen(idx, cap):
    c = torch.from_numpy(np.asarray(canon[idx], np.float32))[None].to(dev, dt)
    u = torch.from_numpy(np.asarray(uv[idx], np.float32))[None].to(dev, dt)
    zz = torch.from_numpy(np.asarray(z[idx], np.float32))[None].to(dev, dt)
    vox = sc(c, u, zz).to(dt)
    e = torch.cat([emb(""), emb(cap)]); voxin = torch.cat([torch.zeros_like(vox), vox])
    g = torch.Generator(dev).manual_seed(a.seed)
    lat = torch.randn(1, 4, 64, 64, generator=g, device=dev, dtype=dt) * sched.init_noise_sigma
    sched.set_timesteps(a.steps, device=dev)
    for t in sched.timesteps:
        li = sched.scale_model_input(torch.cat([lat] * 2), t)
        if a.cond_hi is not None:                                  # t-scheduled: strong early, weak late
            cscale = a.cond_hi if int(t) >= a.cond_t else a.cond_lo
        else:
            cscale = a.cond
        d, mr = cn(li, t, encoder_hidden_states=e, controlnet_cond=voxin,
                   conditioning_scale=cscale, return_dict=False)
        pred = base.unet(li, t, encoder_hidden_states=e, down_block_additional_residuals=d,
                         mid_block_additional_residual=mr, return_dict=False)[0]
        nu, nt = pred.chunk(2)
        lat = sched.step(nu + a.guidance * (nt - nu), t, lat, return_dict=False)[0]
    img = base.vae.decode(lat / base.vae.config.scaling_factor, return_dict=False)[0]
    return ((img / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)[..., ::-1]


rows = []
for k, idx in enumerate(pick):
    cap = "" if a.no_text else man["captions"][idx]
    u = np.asarray(uv[idx], np.float32); zz = np.asarray(z[idx], np.float32)
    rows.append(np.hstack([mesh_ref(u, zz), gen(idx, cap)]))
    print(f"[{k+1}/{a.n}] id={man['ids'][idx]} | {cap[:50]}", flush=True)
os.makedirs("sample", exist_ok=True)
cv2.imwrite(a.out, np.vstack(rows))
print(f"saved {a.out}  (mesh ref | gen, traj100, normal-4500, cfg={a.guidance} cond={a.cond})", flush=True)
