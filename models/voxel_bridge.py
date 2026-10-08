"""Voxel -> ControlNet conditioning embedding (residual 3D encoder + residual 2D head).

Plain-PyTorch, diffusers/MONAI-style ResnetBlock/DownBlock (no monai dependency),
richer than the old shallow bridge so more mesh-voxel information survives.

Flow (depth=32):
  voxel (B, C_in, 128,128, 32)
    conv_in                    -> (B, 64,  128,128, 32)
    DownBlock0 down (2,2,1)    -> (B, 128, 64,64, 32)   # H,W /2
    DownBlock1 down (1,1,2)    -> (B, 192, 64,64, 16)   # D /2
    DownBlock2 down (1,1,2)    -> (B, 256, 64,64, 8)    # D /2
    fold depth into channels   -> (B, 2048, 64,64)
    ResnetBlock2D 2048->1024   -> (B, 1024, 64,64)
    ResnetBlock2D 1024->320    -> (B, 320, 64,64)       # no-op at init
Every block starts as (down-sampled) identity: conv2 is zero-init, and the last
2D head block's channel-reducing skip is zero-init too, so the whole embedding is
0 at init (ControlNet no-op).
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from torch import nn

_CONV = {2: nn.Conv2d, 3: nn.Conv3d}
_POOL = {2: nn.AvgPool2d, 3: nn.AvgPool3d}


def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def _groups(target: int, channels: int) -> int:
    for g in range(min(target, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class Downsample(nn.Module):
    def __init__(self, dims: int, ch: int, use_conv: bool, out_ch: Optional[int] = None, stride=2):
        super().__init__()
        out_ch = out_ch or ch
        if use_conv:
            self.op = _CONV[dims](ch, out_ch, 3, stride=stride, padding=1)
        else:
            assert ch == out_ch, "AvgPool downsample cannot change channels"
            self.op = _POOL[dims](kernel_size=stride, stride=stride)

    def forward(self, x):
        return self.op(x)


class ResnetBlock(nn.Module):
    def __init__(self, dims: int, in_ch: int, out_ch: Optional[int] = None, down: bool = False,
                 down_stride=2, groups: int = 32, zero_skip: bool = False):
        super().__init__()
        out_ch = out_ch or in_ch
        self.act = nn.SiLU()
        self.norm1 = nn.GroupNorm(_groups(groups, in_ch), in_ch, eps=1e-6)
        self.conv1 = _CONV[dims](in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(groups, out_ch), out_ch, eps=1e-6)
        self.conv2 = zero_module(_CONV[dims](out_ch, out_ch, 3, padding=1))
        self.down = Downsample(dims, in_ch, use_conv=False, stride=down_stride) if down else None
        if in_ch != out_ch:
            skip = _CONV[dims](in_ch, out_ch, 1)
            self.skip = zero_module(skip) if zero_skip else skip
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        h = self.act(self.norm1(x))
        if self.down is not None:
            x = self.down(x); h = self.down(h)
        h = self.conv1(h)
        h = self.conv2(self.act(self.norm2(h)))
        return self.skip(x) + h


class DownBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_res_blocks: int = 2,
                 down_stride=None, groups: int = 32):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock(3, in_ch if i == 0 else out_ch, out_ch, groups=groups)
            for i in range(num_res_blocks)])
        self.downsampler = (ResnetBlock(3, out_ch, out_ch, down=True, down_stride=down_stride,
                                        groups=groups) if down_stride is not None else None)

    def forward(self, x):
        for r in self.resnets:
            x = r(x)
        if self.downsampler is not None:
            x = self.downsampler(x)
        return x


class VoxelControlEmbedding(nn.Module):
    """Voxel (B,C,H,W,D) -> ControlNet conditioning (B, cond_ch, H/2, W/2).

    Drop-in:  (in_channels, conditioning_embedding_channels, depth)
    """
    def __init__(
        self,
        in_channels: int = 48,
        conditioning_embedding_channels: int = 320,   # = unet block_out_channels[0]
        conv_in_channels: int = 64,
        down_channels: Sequence[int] = (128, 192, 256),
        depth: int = 32,
        depth_after: int = 8,
        num_res_blocks: int = 2,
        head_channels: Sequence[int] = (1024, 320),    # gradual 2D reduction: 2048 -> 1024 -> 320
        groups: int = 32,
    ):
        super().__init__()
        n_d = int(round(math.log2(depth / depth_after)))          # D halvings (32->8 = 2)
        strides = [(2, 2, 1)] + [(1, 1, 2)] * n_d                  # H,W /2 once, then D /2
        assert len(down_channels) == len(strides), (
            f"need {len(strides)} down_channels for depth {depth}->{depth_after}, got {len(down_channels)}")

        self.conv_in = nn.Conv3d(in_channels, conv_in_channels, 3, padding=1)
        chs = [conv_in_channels] + list(down_channels)
        self.down_blocks = nn.ModuleList([
            DownBlock3D(chs[i], chs[i + 1], num_res_blocks=num_res_blocks,
                        down_stride=strides[i], groups=groups)
            for i in range(len(strides))])

        self.depth_after = depth_after
        fold_ch = down_channels[-1] * depth_after                 # 256*8 = 2048
        assert head_channels[-1] == conditioning_embedding_channels, (
            f"last head channel must be {conditioning_embedding_channels}, got {head_channels[-1]}")

        # 2D residual head: gradual reduction (2048 -> 1024 -> 320). The LAST block is
        # zero-init (skip + conv2) so the whole embedding is 0 at init (ControlNet no-op).
        head, c = [], fold_ch
        n = len(head_channels)
        for i, out_c in enumerate(head_channels):
            head.append(ResnetBlock(2, c, out_c, groups=groups,
                                    zero_skip=(i == n - 1 and c != out_c)))
            c = out_c
        self.head = nn.ModuleList(head)
        self.gradient_checkpointing = False

    def _run(self, blk, x):
        if self.training and self.gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
        return blk(x)

    def forward(self, voxel: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(voxel)
        for blk in self.down_blocks:
            x = self._run(blk, x)                                 # (B, C_last, H/2, W/2, depth_after)
        B, C, H, W, D = x.shape
        x = x.permute(0, 1, 4, 2, 3).reshape(B, C * D, H, W)      # fold depth -> channels
        for blk in self.head:
            x = self._run(blk, x)
        return x                                                  # (B, cond_ch, H/2, W/2)
