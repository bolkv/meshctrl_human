from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.point_transformer import PointEncoder, load_faces_from_obj  # noqa: F401


def vertex_normals(verts: torch.Tensor, faces: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-vertex area-weighted normals. verts (B,N,3), faces (F,3) long (outward winding).
    Returns unit normals (B,N,3). Canonical-frame: front-surface -> +forward, back -> -forward."""
    v0 = verts[:, faces[:, 0]]; v1 = verts[:, faces[:, 1]]; v2 = verts[:, faces[:, 2]]  # (B,F,3)
    fn = torch.cross(v1 - v0, v2 - v0, dim=-1)                      # (B,F,3) area-weighted face normal
    vn = torch.zeros_like(verts)
    idx = torch.cat([faces[:, 0], faces[:, 1], faces[:, 2]], dim=0)  # (3F,)
    vn.index_add_(1, idx, fn.repeat(1, 3, 1))                       # scatter face normal to its 3 verts
    return F.normalize(vn, dim=-1, eps=eps)


class FourierPositionalEncoding(nn.Module):
    def __init__(self, num_freqs: int = 4, include_input: bool = True, base: float = 2.0):
        super().__init__()
        self.include_input = include_input
        freqs = (base ** torch.arange(num_freqs)) * math.pi
        self.register_buffer("freqs", freqs, persistent=False)
        self.out_dim = 3 * (int(include_input) + 2 * num_freqs)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        out = [coords] if self.include_input else []
        scaled = coords.unsqueeze(-1) * self.freqs
        out.append(scaled.sin().flatten(-2))
        out.append(scaled.cos().flatten(-2))
        return torch.cat(out, dim=-1)


def camera_voxel_scatter(
    cam_coords: torch.Tensor,    # (B, N, 3) camera-space coords (cam_R @ V + cam_T)
    feats: torch.Tensor,         # (B, N, C) per-vertex features
    H: int = 128,
    W: int = 128,
    D: int = 32,
    bounds: Optional[torch.Tensor] = None,
    margin: float = 0.02,
    eps: float = 1e-5,
) -> torch.Tensor:
    B, N, C = feats.shape
    M = H * W * D
    device = feats.device

    if bounds is None:
        cmin = cam_coords.amin(dim=1, keepdim=True)         # (B,1,3)
        cmax = cam_coords.amax(dim=1, keepdim=True)
    else:
        if bounds.dim() == 2:
            bounds = bounds.unsqueeze(0)                    # (1,2,3)
        cmin, cmax = bounds[:, 0:1, :], bounds[:, 1:2, :]

    center = 0.5 * (cmin + cmax)                            # (B,1,3)
    scale = (cmax - cmin).amax(dim=-1, keepdim=True).clamp_min(1e-6)  # (B,1,1) single scale
    n = (cam_coords - center) / scale + 0.5                 # centered, aspect-true
    n = n * (1.0 - 2 * margin) + margin

    res = cam_coords.new_tensor([W - 1, H - 1, D - 1])      # x->W, y->H, z->D
    g = n.clamp(0.0, 1.0) * res
    gx, gy, gz = g[..., 0], g[..., 1], g[..., 2]

    x0 = torch.floor(gx); fx = gx - x0
    y0 = torch.floor(gy); fy = gy - y0
    z0 = torch.floor(gz); fz = gz - z0
    x0 = x0.long(); y0 = y0.long(); z0 = z0.long()

    acc_feat = torch.zeros(B, M, C, device=device, dtype=feats.dtype)
    acc_w = torch.zeros(B, M, 1, device=device, dtype=feats.dtype)

    for dx in (0, 1):
        wx = fx if dx else (1.0 - fx)
        xi = (x0 + dx).clamp(0, W - 1)
        for dy in (0, 1):
            wy = fy if dy else (1.0 - fy)
            yi = (y0 + dy).clamp(0, H - 1)
            for dz in (0, 1):
                wz = fz if dz else (1.0 - fz)
                zi = (z0 + dz).clamp(0, D - 1)

                w = (wx * wy * wz).unsqueeze(-1)              # (B, N, 1)
                flat = (yi * W + xi) * D + zi                 # (B, N) image-natural
                idx_c = flat.unsqueeze(-1).expand(-1, -1, C)
                acc_feat.scatter_add_(1, idx_c, feats * w)
                acc_w.scatter_add_(1, flat.unsqueeze(-1), w)

    feat_grid = acc_feat / acc_w.clamp_min(eps)             # (B, M, C) weighted average
    grid = feat_grid.view(B, H, W, D, C).permute(0, 4, 1, 2, 3).contiguous()
    return grid                                             # (B, C, H, W, D)


def frustum_trilinear_scatter(
    uv: torch.Tensor,            # (B, N, 2) projected image coords in [0, img_size]
    z: torch.Tensor,             # (B, N)    camera depth
    feats: torch.Tensor,         # (B, N, C) per-vertex features
    H: int = 128,
    W: int = 128,
    D: int = 32,
    img_size: int = 512,
    eps: float = 1e-5,
    drop_oob: bool = False,      # True: drop verts whose (u,v) leave the frame (vs clamp to border)
) -> torch.Tensor:
    B, N, C = feats.shape
    M = H * W * D
    device = feats.device

    uv = torch.nan_to_num(uv, nan=0.0, posinf=2.0 * img_size, neginf=-1.0 * img_size)
    z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    gx = uv[..., 0] / img_size * W                          # -> W (cols)
    gy = uv[..., 1] / img_size * H                          # -> H (rows)
    zmin = z.amin(dim=1, keepdim=True)
    zmax = z.amax(dim=1, keepdim=True)
    gz = (z - zmin) / (zmax - zmin + 1e-6) * (D - 1)        # per-sample depth

    x0 = torch.floor(gx); fx = gx - x0
    y0 = torch.floor(gy); fy = gy - y0
    z0 = torch.floor(gz); fz = gz - z0
    x0 = x0.long(); y0 = y0.long(); z0 = z0.long()

    acc_feat = torch.zeros(B, M, C, device=device, dtype=feats.dtype)
    acc_w = torch.zeros(B, M, 1, device=device, dtype=feats.dtype)

    for dx in (0, 1):
        wx = fx if dx else (1.0 - fx)
        xraw = x0 + dx; xi = xraw.clamp(0, W - 1)
        okx = (xraw >= 0) & (xraw <= W - 1)
        for dy in (0, 1):
            wy = fy if dy else (1.0 - fy)
            yraw = y0 + dy; yi = yraw.clamp(0, H - 1)
            oky = (yraw >= 0) & (yraw <= H - 1)
            for dz in (0, 1):
                wz = fz if dz else (1.0 - fz)
                zi = (z0 + dz).clamp(0, D - 1)
                w = wx * wy * wz
                if drop_oob:                       # out-of-frame x/y corner -> zero weight
                    w = w * (okx & oky).to(w.dtype)
                w = w.unsqueeze(-1)
                flat = (yi * W + xi) * D + zi
                idx_c = flat.unsqueeze(-1).expand(-1, -1, C)
                acc_feat.scatter_add_(1, idx_c, feats * w)
                acc_w.scatter_add_(1, flat.unsqueeze(-1), w)

    # feat_grid = acc_feat / acc_w.clamp_min(eps)          # (old: weighted average - normalized away density)
    feat_grid = acc_feat                                   # sum (no /acc_w): keep density/count info
    return feat_grid.view(B, H, W, D, C).permute(0, 4, 1, 2, 3).contiguous()


# ShapeCtrl encoder
class ShapeCtrl(nn.Module):
    def __init__(
        self,
        faces: Optional[torch.Tensor] = None,  # (F, 3) long, 0-indexed (fixed topology)
        num_vertices: int = 10475,           # SMPL-X template vertex count
        fourier_num_freqs: int = 4,          # -> 27-D PE
        pt_dims: Sequence[int] = (54, 48, 48, 48),   # in=54 (canon PE 27 + normal PE 27); 3 PT layers
        voxel_res: Sequence[int] = (128, 128, 32),   # (H, W, D)
        max_neighbors: Optional[int] = 8,    # cap 1-ring (fan poles only) to save memory
        adjacency_cache: Optional[str] = None,  # build once, reuse for all meshes
        drop_oob: bool = False,              # drop out-of-frame verts in scatter (vs clamp)
        faces_path: str = "data/smplx_faces.npy",  # for per-vertex normals
    ):
        super().__init__()
        self.num_vertices = num_vertices
        self.H, self.W, self.D = tuple(voxel_res)
        self.drop_oob = drop_oob

        # faces (for canonical-frame vertex normals): encode front/back surface orientation
        if faces is None:
            faces = torch.as_tensor(np.load(faces_path), dtype=torch.long)
        self.register_buffer("faces", faces.long(), persistent=False)

        # Fourier PE of canonical coords AND of canonical vertex normal (facing signal)
        self.pe = FourierPositionalEncoding(num_freqs=fourier_num_freqs, include_input=True)
        self.pe_n = FourierPositionalEncoding(num_freqs=fourier_num_freqs, include_input=True)
        in_dim = self.pe.out_dim + self.pe_n.out_dim        # 27 + 27 = 54
        assert pt_dims[0] == in_dim, (
            f"pt_dims[0] ({pt_dims[0]}) must equal canon PE + normal PE dim ({in_dim})"
        )

        # 4. point encoder over fixed mesh connectivity: 54 -> 48 -> 48 -> 48
        self.point_encoder = PointEncoder(
            faces, num_vertices, dims=tuple(pt_dims),
            max_neighbors=max_neighbors, adjacency_cache=adjacency_cache,
        )
        self.feat_dim = self.point_encoder.out_dim         # 24
        self.out_channels = self.feat_dim                  # 24

    def forward(
        self,
        canon_coords: torch.Tensor,
        uv: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """canon_coords: (B,N,3) for encoding; uv: (B,N,2) projected image coords;
        z: (B,N) camera depth. Returns (B, 24, H, W, D) frustum voxel."""
        B, N, _ = canon_coords.shape

        # 1. per-vertex canonical normal (facing signal): front-surface vs back-surface
        normals = vertex_normals(canon_coords, self.faces)  # (B, N, 3) unit

        # 2. Fourier PE of canonical coords + Fourier PE of normal
        pe = self.pe(canon_coords)                          # (B, N, 27)
        pe_n = self.pe_n(normals)                           # (B, N, 27)

        # 3. concat -> 54-D
        x = torch.cat([pe, pe_n], dim=-1)                   # (B, N, 54)

        # 4. point encoder over fixed mesh connectivity (pos = canonical) -> (B, N, 24)
        x = self.point_encoder(x, canon_coords)

        # 5. frustum scatter at projected (u,v) + depth -> image-aligned voxel
        voxel = frustum_trilinear_scatter(
            uv, z, x, H=self.H, W=self.W, D=self.D, drop_oob=self.drop_oob,
        )
        return voxel                                        # (B, 24, H, W, D)
