"""Sample from the trained single-body ShapeCtrl + ControlNet on bedlam_single test samples.

For N random test samples: ShapeCtrl(canon,uv,z) -> voxel -> ControlNet residuals,
caption -> text encoder, SD1.5 UNet denoising (CFG) -> VAE decode.
Saves a grid: each row = [GT crop | generated].
"""
import os, json, random, argparse, numpy as np, torch, cv2
from diffusers import UniPCMultistepScheduler
from safetensors.torch import load_file
from configs.model_config import get_config
from models.controlnet import attach_voxel_bridge, build_controlnet, load_base_models
from models.shapectrl import ShapeCtrl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/shapectrl_bedlam_single/final")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--cond", type=float, default=0.7)       # controlnet conditioning_scale
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="sample/bedlam_test_montage.jpg")
    args = ap.parse_args()

    cfg = get_config([]); dev = "cuda"; dt = torch.float16
    man = json.load(open(f"{cfg.single_dir}/manifest.json"))
    idxs = man["test_idx"] if args.split == "test" else man["train_idx"]
    pick = random.Random(args.seed).sample(idxs, args.n)

    base = load_base_models(cfg)
    shapectrl = ShapeCtrl(num_vertices=cfg.num_vertices, adjacency_cache=cfg.adjacency_cache, drop_oob=cfg.drop_oob)
    controlnet = build_controlnet(cfg, base.unet)
    attach_voxel_bridge(controlnet, feat_dim=shapectrl.out_channels, depth=shapectrl.D)
    controlnet.load_state_dict(load_file(f"{args.ckpt}/controlnet/diffusion_pytorch_model.safetensors"))
    shapectrl.load_state_dict(torch.load(f"{args.ckpt}/shapectrl.pt", map_location="cpu"))
    for m in (base.vae, base.unet, base.text_encoder, controlnet): m.to(dev, dt).eval()
    shapectrl.to(dev).float().eval()
    sched = UniPCMultistepScheduler.from_config(base.noise_scheduler.config)
    tok = base.tokenizer

    def embed(text):
        ids = tok(text, max_length=tok.model_max_length, padding="max_length",
                  truncation=True, return_tensors="pt").input_ids.to(dev)
        return base.text_encoder(ids, return_dict=False)[0]

    rows = []
    for k, j in enumerate(pick):
        d = os.path.join(cfg.single_dir, man["samples"][j])
        z = np.load(f"{d}/meta.npz", allow_pickle=True)
        canon = torch.tensor(z["canon"].astype(np.float32))[None].to(dev)
        uv = torch.tensor(z["uv"].astype(np.float32))[None].to(dev)
        zz = torch.tensor(z["z"].astype(np.float32))[None].to(dev)
        cap = open(f"{d}/caption.txt", encoding="utf-8").read().strip()
        with torch.no_grad():
            sched.set_timesteps(args.steps, device=dev)
            text_emb = torch.cat([embed(""), embed(cap)]).to(dt)
            voxel = shapectrl(canon, uv, zz).to(dt); voxel = torch.cat([voxel, voxel])
            g = torch.Generator(dev).manual_seed(args.seed + k)
            lat = torch.randn(1, 4, 64, 64, generator=g, device=dev, dtype=dt) * sched.init_noise_sigma
            for t in sched.timesteps:
                li = sched.scale_model_input(torch.cat([lat, lat]), t)
                down, mid = controlnet(li, t, encoder_hidden_states=text_emb,
                                       controlnet_cond=voxel, conditioning_scale=args.cond,
                                       return_dict=False)
                pred = base.unet(li, t, encoder_hidden_states=text_emb,
                                 down_block_additional_residuals=down,
                                 mid_block_additional_residual=mid, return_dict=False)[0]
                nu, nc = pred.chunk(2); pred = nu + args.guidance * (nc - nu)
                lat = sched.step(pred, t, lat, return_dict=False)[0]
            img = base.vae.decode(lat / base.vae.config.scaling_factor, return_dict=False)[0]
        gen = ((img / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
        gt = cv2.cvtColor(cv2.imread(f"{d}/img.jpg"), cv2.COLOR_BGR2RGB)
        rows.append(np.hstack([gt, cv2.resize(gen, (512, 512))]))     # GT|gen pair (1024x512)
        print(f"[{k+1}/{args.n}] {man['samples'][j]} | {cap[:55]}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cols = 3; nrow = (len(rows) + cols - 1) // cols; h, w = rows[0].shape[:2]
    grid = np.full((nrow * h, cols * w, 3), 255, np.uint8)
    for i, r in enumerate(rows):
        rr, cc = divmod(i, cols); grid[rr*h:(rr+1)*h, cc*w:(cc+1)*w] = r
    cv2.imwrite(args.out, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"saved {args.out}  (각 칸: 좌=GT, 우=생성 | cfg={args.guidance} cond={args.cond})", flush=True)


if __name__ == "__main__":
    main()
