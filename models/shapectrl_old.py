"""Old (pre-normal) ShapeCtrl: vertex_embed(21) + FourierPE(canon)(27) = 48, no normals.
Reuses scatter + PE from the current shapectrl module; only the encoder head differs."""
from typing import Optional, Sequence
import torch
from torch import nn

from models.point_transformer import PointEncoder
from models.shapectrl import FourierPositionalEncoding, frustum_trilinear_scatter


class ShapeCtrlOld(nn.Module):
    def __init__(self, faces=None, num_vertices=10475, fourier_num_freqs=4,
                 pt_dims: Sequence[int] = (48, 48, 48, 48), voxel_res=(128, 128, 32),
                 max_neighbors=8, adjacency_cache=None, drop_oob=False):
        super().__init__()
        self.num_vertices = num_vertices
        self.H, self.W, self.D = tuple(voxel_res)
        self.drop_oob = drop_oob
        self.pe = FourierPositionalEncoding(num_freqs=fourier_num_freqs, include_input=True)
        embed_dim = pt_dims[0] - self.pe.out_dim                      # 48 - 27 = 21
        self.vertex_embed = nn.Embedding(num_vertices, embed_dim)
        self.point_encoder = PointEncoder(faces, num_vertices, dims=tuple(pt_dims),
                                          max_neighbors=max_neighbors, adjacency_cache=adjacency_cache)
        self.feat_dim = self.point_encoder.out_dim
        self.out_channels = self.feat_dim

    def forward(self, canon_coords, uv, z):
        B, N, _ = canon_coords.shape
        ids = torch.arange(N, device=canon_coords.device)
        embed = self.vertex_embed(ids).unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([embed, self.pe(canon_coords)], dim=-1)
        x = self.point_encoder(x, canon_coords)
        return frustum_trilinear_scatter(uv, z, x, H=self.H, W=self.W, D=self.D, drop_oob=self.drop_oob)
