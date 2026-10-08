from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from torch import nn

from models.point_transformer import PointEncoder, load_faces_from_obj  # noqa: F401



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

    feat_grid = acc_feat / acc_w.clamp_min(eps)            # weighted average over contributions
    return feat_grid.view(B, H, W, D, C).permute(0, 4, 1, 2, 3).contiguous()


# ShapeCtrl encoder
class ShapeCtrl(nn.Module):
    def __init__(
        self,
        faces: Optional[torch.Tensor] = None,  # (F, 3) long, 0-indexed (fixed topology)
        num_vertices: int = 10475,           # SMPL-X template vertex count
        fourier_num_freqs: int = 4,          # -> 27-D PE
        pt_dims: Sequence[int] = (48, 48, 48, 48),   # in=48 (vertex embed 21 + canon PE 27)
        voxel_res: Sequence[int] = (128, 128, 32),   # (H, W, D)
        max_neighbors: Optional[int] = 8,    # cap 1-ring (fan poles only) to save memory
        adjacency_cache: Optional[str] = None,  # build once, reuse for all meshes
        drop_oob: bool = False,              # drop out-of-frame verts in scatter (vs clamp)
    ):
        super().__init__()
        self.num_vertices = num_vertices
        self.H, self.W, self.D = tuple(voxel_res)
        self.drop_oob = drop_oob

        # Fourier PE; the vertex index embedding fills the rest of pt_dims[0]
        self.pe = FourierPositionalEncoding(num_freqs=fourier_num_freqs, include_input=True)
        pe_dim = self.pe.out_dim                            # 27
        embed_dim = pt_dims[0] - pe_dim                     # 48 - 27 = 21
        assert embed_dim > 0, (
            f"pt_dims[0] ({pt_dims[0]}) must exceed Fourier PE dim ({pe_dim})"
        )

        # per-vertex index embedding, used directly (no refine MLP)
        self.vertex_embed = nn.Embedding(num_vertices, embed_dim)   # (N, 21)

        # point encoder over fixed mesh connectivity: 48 -> 48 -> 48 -> 48
        self.point_encoder = PointEncoder(
            faces, num_vertices, dims=tuple(pt_dims),
            max_neighbors=max_neighbors, adjacency_cache=adjacency_cache,
        )
        self.feat_dim = self.point_encoder.out_dim
        self.out_channels = self.feat_dim

    def forward(
        self,
        canon_coords: torch.Tensor,
        uv: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """canon_coords: (B,N,3) for encoding; uv: (B,N,2) projected image coords;
        z: (B,N) camera depth. Returns (B, out_channels, H, W, D) frustum voxel."""
        B, N, _ = canon_coords.shape

        # 1. per-vertex index embedding (direct, no refine MLP)
        vertex_ids = torch.arange(N, device=canon_coords.device)
        embed = self.vertex_embed(vertex_ids).unsqueeze(0).expand(B, -1, -1)  # (B, N, 21)

        # 2. Fourier PE of canonical coords
        pe = self.pe(canon_coords)                          # (B, N, 27)

        # 3. concat -> 48-D
        x = torch.cat([embed, pe], dim=-1)                  # (B, N, 48)

        # 4. point encoder over fixed mesh connectivity (pos = canonical)
        x = self.point_encoder(x, canon_coords)

        # 5. frustum scatter at projected (u,v) + depth -> image-aligned voxel
        voxel = frustum_trilinear_scatter(
            uv, z, x, H=self.H, W=self.W, D=self.D, drop_oob=self.drop_oob,
        )
        return voxel                                        # (B, out_channels, H, W, D)
