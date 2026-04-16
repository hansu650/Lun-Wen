"""DFormer-inspired RGB-D model for the local teaching framework."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_lit import BaseLitSeg
from .decoder import SimpleFPNDecoder


class ConvMLP(nn.Module):
    """A small Conv-MLP block inspired by the MLP block in DFormer."""

    def __init__(self, channels, mlp_ratio=4):
        super().__init__()
        hidden = channels * mlp_ratio
        self.norm = nn.BatchNorm2d(channels)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        # 用 depthwise conv 给通道混合加一点局部位置感，这个思路是借 DFormer 的 pos conv。
        self.pos = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.pos(x) + x
        x = self.act(x)
        x = self.fc2(x)
        return x


class DFormerInteractionBlock(nn.Module):
    """
    A lightweight RGB-depth interaction block inspired by DFormer.

    这里不是官方 DFormer 的完整实现，而是借它几个最核心的结构习惯：
    1. RGB / Depth 双分支同时走
    2. 每个 stage 内都让两种模态持续交互，而不是只在某一层融合一次
    3. 用局部卷积 + 门控交互替代完整的官方复杂注意力
    """

    def __init__(self, rgb_channels, depth_channels, mlp_ratio=4):
        super().__init__()
        self.rgb_norm = nn.BatchNorm2d(rgb_channels)
        self.depth_norm = nn.BatchNorm2d(depth_channels)

        # RGB / Depth 各自先做一层 depthwise conv，模拟 DFormer 里“局部建模 + 轻量交互”的味道。
        self.rgb_local = nn.Conv2d(
            rgb_channels, rgb_channels, kernel_size=5, padding=2, groups=rgb_channels
        )
        self.depth_local = nn.Conv2d(
            depth_channels, depth_channels, kernel_size=5, padding=2, groups=depth_channels
        )

        # 两个方向的投影：
        # depth -> rgb 用来给 RGB 提供几何提示；
        # rgb -> depth 用来让 depth 分支也保留一点视觉语义。
        self.depth_to_rgb = nn.Conv2d(depth_channels, rgb_channels, kernel_size=1)
        self.rgb_to_depth = nn.Conv2d(rgb_channels, depth_channels, kernel_size=1)

        # 门控权重不直接拿单个分支自己算，而是看 RGB + Depth 的拼接结果，
        # 这样更符合“跨模态交互”这个目标。
        self.rgb_gate = nn.Sequential(
            nn.Conv2d(rgb_channels * 2, rgb_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.depth_gate = nn.Sequential(
            nn.Conv2d(depth_channels * 2, depth_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        self.rgb_fuse = nn.Sequential(
            nn.Conv2d(rgb_channels * 2, rgb_channels, kernel_size=1),
            nn.BatchNorm2d(rgb_channels),
            nn.GELU(),
        )
        self.depth_fuse = nn.Sequential(
            nn.Conv2d(depth_channels * 2, depth_channels, kernel_size=1),
            nn.BatchNorm2d(depth_channels),
            nn.GELU(),
        )

        self.rgb_mlp = ConvMLP(rgb_channels, mlp_ratio=mlp_ratio)
        self.depth_mlp = ConvMLP(depth_channels, mlp_ratio=mlp_ratio)

    def forward(self, rgb, depth):
        rgb_res = rgb
        depth_res = depth

        rgb = self.rgb_norm(rgb)
        depth = self.depth_norm(depth)

        rgb_local = self.rgb_local(rgb)
        depth_local = self.depth_local(depth)

        depth_hint = self.depth_to_rgb(depth_local)
        rgb_gate = self.rgb_gate(torch.cat([rgb_local, depth_hint], dim=1))
        rgb_update = self.rgb_fuse(torch.cat([rgb_local, depth_hint * rgb_gate], dim=1))

        rgb_hint = self.rgb_to_depth(rgb_local)
        depth_gate = self.depth_gate(torch.cat([depth_local, rgb_hint], dim=1))
        depth_update = self.depth_fuse(torch.cat([depth_local, rgb_hint * depth_gate], dim=1))

        # 先做一层跨模态更新，再接一层 Conv-MLP。
        rgb = rgb_res + rgb_update
        depth = depth_res + depth_update
        rgb = rgb + self.rgb_mlp(rgb)
        depth = depth + self.depth_mlp(depth)

        return rgb, depth


class DFormerStage(nn.Module):
    """One stage = a few repeated interaction blocks."""

    def __init__(self, rgb_channels, depth_channels, depth=2, mlp_ratio=4):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                DFormerInteractionBlock(
                    rgb_channels=rgb_channels,
                    depth_channels=depth_channels,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(depth)
            ]
        )
        self.depth_to_rgb_out = nn.Conv2d(depth_channels, rgb_channels, kernel_size=1)

    def forward(self, rgb, depth):
        for block in self.blocks:
            rgb, depth = block(rgb, depth)

        # 给 decoder 的输出仍然只保留 RGB 主分支，
        # 但在输出前把 depth 分支投影回 RGB 通道，相当于“带着几何增强的 RGB 特征”。
        fused_out = rgb + self.depth_to_rgb_out(depth)
        return rgb, depth, fused_out


class DFormerInspiredEncoder(nn.Module):
    """
    Simplified DFormer-like encoder.

    借鉴点：
    - RGB / Depth 双分支 stem
    - 每个 stage 都做持续交互
    - Depth 分支通道数通常比 RGB 分支更小
    - 最后输出多尺度特征给 FPN decoder
    """

    def __init__(self, depths=(2, 2, 4, 2), dims=(64, 128, 256, 512)):
        super().__init__()
        self.out_channels = list(dims)
        depth_dims = [dim // 2 for dim in dims]

        # RGB stem：和 DFormer 一样，先两次 stride=2 下采样，快速到 1/4 分辨率。
        self.rgb_stem = nn.Sequential(
            nn.Conv2d(3, dims[0] // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dims[0] // 2),
            nn.GELU(),
            nn.Conv2d(dims[0] // 2, dims[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.GELU(),
        )

        # Depth stem：思路和官方 DFormer 接近，但通道数更轻。
        self.depth_stem = nn.Sequential(
            nn.Conv2d(1, depth_dims[0] // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(depth_dims[0] // 2),
            nn.GELU(),
            nn.Conv2d(depth_dims[0] // 2, depth_dims[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(depth_dims[0]),
            nn.GELU(),
        )

        self.rgb_downsamples = nn.ModuleList()
        self.depth_downsamples = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.rgb_downsamples.append(
                nn.Sequential(
                    nn.BatchNorm2d(dims[i]),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=3, stride=2, padding=1),
                )
            )
            self.depth_downsamples.append(
                nn.Sequential(
                    nn.BatchNorm2d(depth_dims[i]),
                    nn.Conv2d(depth_dims[i], depth_dims[i + 1], kernel_size=3, stride=2, padding=1),
                )
            )

        self.stages = nn.ModuleList(
            [
                DFormerStage(
                    rgb_channels=dims[i],
                    depth_channels=depth_dims[i],
                    depth=depths[i],
                    mlp_ratio=4 if i < 2 else 3,
                )
                for i in range(len(dims))
            ]
        )

    def forward(self, rgb, depth):
        rgb = self.rgb_stem(rgb)
        depth = self.depth_stem(depth)

        feats = []
        for i, stage in enumerate(self.stages):
            rgb, depth, fused = stage(rgb, depth)
            feats.append(fused)
            if i < len(self.stages) - 1:
                rgb = self.rgb_downsamples[i](rgb)
                depth = self.depth_downsamples[i](depth)
        return feats


class DFormerInspiredSegmentor(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()
        self.encoder = DFormerInspiredEncoder()
        self.decoder = SimpleFPNDecoder(self.encoder.out_channels, num_classes=num_classes)

    def forward(self, rgb, depth):
        feats = self.encoder(rgb, depth)
        return self.decoder(feats, input_size=rgb.shape[-2:])


class LitDFormerInspired(BaseLitSeg):
    def __init__(self, num_classes=40, lr=1e-4):
        super().__init__(num_classes=num_classes, lr=lr)
        # 这里的名字叫 DFormerInspired，
        # 是为了明确告诉你：它借鉴了 DFormer 的双分支持续交互思路，
        # 但不是官方仓库那版完整 DFormer / DFormerv2。
        self.model = DFormerInspiredSegmentor(num_classes=num_classes)
