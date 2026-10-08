import torch
from torch import nn

# helpers

def exists(val):
    return val is not None

def max_value(t):
    return torch.finfo(t.dtype).max

def load_faces_from_obj(path):
    """Read faces from an .obj. Returns (F, 3) long, 0-indexed."""
    faces = []
    with open(path, "r") as f:
        for line in f:
            if not line.startswith("f "):
                continue
            # handle "f v", "f v/vt", "f v/vt/vn"
            faces.append([int(tok.split("/")[0]) for tok in line.split()[1:4]])
    return torch.tensor(faces, dtype=torch.long) - 1  # obj is 1-indexed


def build_vertex_adjacency(faces, num_vertices, include_self=True, max_neighbors=None):
    e = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0)
    e = torch.cat([e, e.flip(1)], dim=0)  # undirected

    neighbors = [set() for _ in range(num_vertices)]
    for s, d in e.tolist():
        neighbors[s].add(d)

    # self always first so it survives truncation
    lists = []
    for i in range(num_vertices):
        others = sorted(neighbors[i] - {i})
        lst = ([i] + others) if include_self else others
        lists.append(lst)

    K = max(len(l) for l in lists)
    if max_neighbors is not None:
        K = min(K, max_neighbors)

    neighbor_idx = torch.zeros(num_vertices, K, dtype=torch.long)
    neighbor_mask = torch.zeros(num_vertices, K, dtype=torch.bool)
    for i, l in enumerate(lists):
        l = l[:K]
        neighbor_idx[i, : len(l)] = torch.tensor(l, dtype=torch.long)
        neighbor_mask[i, : len(l)] = True
    return neighbor_idx, neighbor_mask


def load_or_build_adjacency(
    faces=None, num_vertices=None, cache_path=None, include_self=True, max_neighbors=None
):

    import os

    if cache_path is not None and os.path.exists(cache_path):
        d = torch.load(cache_path)
        return d["neighbor_idx"], d["neighbor_mask"]

    if faces is None:
        raise ValueError("No cache found and no `faces` given to build adjacency.")
    neighbor_idx, neighbor_mask = build_vertex_adjacency(
        faces, num_vertices, include_self=include_self, max_neighbors=max_neighbors
    )
    if cache_path is not None:
        torch.save({"neighbor_idx": neighbor_idx, "neighbor_mask": neighbor_mask}, cache_path)
    return neighbor_idx, neighbor_mask


def gather_neighbors(feat, idx):
    """feat: (B, N, D), idx: (N, K) -> (B, N, K, D)."""
    B, N, D = feat.shape
    K = idx.shape[1]
    idx_e = idx.reshape(1, N * K, 1).expand(B, N * K, D)
    return torch.gather(feat, 1, idx_e).view(B, N, K, D)


class PointTransformerLayer(nn.Module):

    def __init__(
        self,
        *,
        dim,
        pos_mlp_hidden_dim = 64,
        attn_mlp_hidden_mult = 4,
    ):
        super().__init__()
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.pos_mlp = nn.Sequential(
            nn.Linear(3, pos_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(pos_mlp_hidden_dim, dim)
        )

        self.attn_mlp = nn.Sequential(
            nn.Linear(dim, dim * attn_mlp_hidden_mult),
            nn.ReLU(),
            nn.Linear(dim * attn_mlp_hidden_mult, dim),
        )

    def forward(self, x, pos, neighbor_idx, neighbor_mask):
        # x: (B, N, d) | pos: (B, N, 3) | neighbor_idx/mask: (N, K)
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # gather only the fixed-connectivity neighbours
        k_n = gather_neighbors(k, neighbor_idx)        # (B, N, K, d)
        v_n = gather_neighbors(v, neighbor_idx)        # (B, N, K, d)
        pos_n = gather_neighbors(pos, neighbor_idx)    # (B, N, K, 3)

        # relative positional embeddings
        rel_pos = pos[:, :, None, :] - pos_n           # (B, N, K, 3)
        rel_pos_emb = self.pos_mlp(rel_pos)            # (B, N, K, d)

        # subtraction attention (better inductive bias for points than dot product)
        qk_rel = q[:, :, None, :] - k_n                # (B, N, K, d)

        # add positional embedding before the attention mlp
        v = v_n + rel_pos_emb
        sim = self.attn_mlp(qk_rel + rel_pos_emb)      # (B, N, K, d)

        # mask out padded neighbours
        mask = neighbor_mask[None, :, :, None]         # (1, N, K, 1)
        sim = sim.masked_fill(~mask, -max_value(sim))

        # attention over the K neighbours
        attn = sim.softmax(dim = -2)

        # aggregate
        agg = (attn * v).sum(dim = 2)                  # (B, N, d)
        return agg

class PointTransformerBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out) if dim_in != dim_out else nn.Identity()
        self.norm = nn.LayerNorm(dim_out)
        self.attn = PointTransformerLayer(dim = dim_out)

    def forward(self, x, pos, neighbor_idx, neighbor_mask):
        x = self.proj(x)
        x = x + self.attn(self.norm(x), pos, neighbor_idx, neighbor_mask)
        return x


class PointEncoder(nn.Module):

    def __init__(
        self,
        faces = None,
        num_vertices = None,
        dims = (48, 48, 24),
        include_self = True,
        max_neighbors = None,
        adjacency_cache = None,
    ):
        super().__init__()
        neighbor_idx, neighbor_mask = load_or_build_adjacency(
            faces, num_vertices, cache_path = adjacency_cache,
            include_self = include_self, max_neighbors = max_neighbors,
        )
        # registered (non-persistent) so they follow .to()/.cuda() but aren't saved
        self.register_buffer("neighbor_idx", neighbor_idx, persistent = False)
        self.register_buffer("neighbor_mask", neighbor_mask, persistent = False)

        self.blocks = nn.ModuleList(
            [PointTransformerBlock(d_in, d_out) for d_in, d_out in zip(dims[:-1], dims[1:])]
        )
        self.in_dim = dims[0]
        self.out_dim = dims[-1]

    def forward(self, x, pos):
        for block in self.blocks:
            x = block(x, pos, self.neighbor_idx, self.neighbor_mask)
        return x
