"""Mid-fusion model with a KTB-style cross-modal block on c3/c4."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_lit import BaseLitSeg
from .decoder import SimpleFPNDecoder
from .encoder import DepthEncoder, RGBEncoder


class GatedFusion(nn.Module):
    """Simple low-level fusion kept from the original project.

    We intentionally keep c1/c2 lightweight and stable, because low-level
    features are still mostly about edges and texture. The KTB-style block is
    only inserted on c3/c4 where semantic interaction matters more.
    """

    def __init__(self, rgb_channels, depth_channels):
        super().__init__()
        self.depth_proj = nn.Conv2d(depth_channels, rgb_channels, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(rgb_channels * 2, rgb_channels, 1, bias=False),
            nn.BatchNorm2d(rgb_channels),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(rgb_channels, rgb_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(rgb_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb_feat, depth_feat):
        depth_feat = self.depth_proj(depth_feat)
        gate = self.gate(torch.cat([rgb_feat, depth_feat], dim=1))
        fused = gate * rgb_feat + (1.0 - gate) * depth_feat
        return self.refine(fused)


class KTBChannelWeights(nn.Module):
    """KTB FRM-style channel reweighting.

    The idea comes from KTB's FeatureRectifyModule: use both average/max pooled
    summary statistics from RGB and depth together, then predict how much one
    modality should补给 the other modality along the channel dimension.
    """

    def __init__(self, dim, reduction=4):
        super().__init__()
        hidden_dim = max((dim * 4) // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim * 2),
            nn.Sigmoid(),
        )

    def forward(self, rgb_feat, depth_feat):
        batch_size, channels, _, _ = rgb_feat.shape
        merged = torch.cat([rgb_feat, depth_feat], dim=1)
        avg = self.avg_pool(merged).view(batch_size, -1)
        maxv = self.max_pool(merged).view(batch_size, -1)
        weights = self.mlp(torch.cat([avg, maxv], dim=1)).view(batch_size, 2, channels, 1, 1)
        return weights[:, 0], weights[:, 1]


class KTBSpatialWeights(nn.Module):
    """KTB FRM-style spatial reweighting."""

    def __init__(self, dim, reduction=4):
        super().__init__()
        hidden_dim = max(dim // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 2, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 2, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, rgb_feat, depth_feat):
        weights = self.mlp(torch.cat([rgb_feat, depth_feat], dim=1))
        return weights[:, 0:1], weights[:, 1:2]


class KTBFeatureRectify(nn.Module):
    """Rectify RGB/depth before token-level interaction.

    This is the closest small reusable piece to KTB's FRM idea. It preserves
    NCHW shape and only reweights the counterpart modality before fusion.
    """

    def __init__(self, dim, reduction=4, lambda_c=0.5, lambda_s=0.5):
        super().__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weights = KTBChannelWeights(dim, reduction=reduction)
        self.spatial_weights = KTBSpatialWeights(dim, reduction=reduction)

    def forward(self, rgb_feat, depth_feat):
        rgb_channel, depth_channel = self.channel_weights(rgb_feat, depth_feat)
        rgb_spatial, depth_spatial = self.spatial_weights(rgb_feat, depth_feat)

        rgb_rect = rgb_feat + self.lambda_c * depth_channel * depth_feat + self.lambda_s * depth_spatial * depth_feat
        depth_rect = depth_feat + self.lambda_c * rgb_channel * rgb_feat + self.lambda_s * rgb_spatial * rgb_feat
        return rgb_rect, depth_rect


class KTBCrossAttention(nn.Module):
    """Token interaction adapted from KTB's CrossAttention/CrossPath path.

    Input and output both stay in (B, N, C), so we can flatten c3/c4 feature
    maps, exchange context, and then reshape back without touching decoder code.
    """

    def __init__(self, dim, num_heads=4):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"Reduced dim {dim} must be divisible by heads {num_heads}.")

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.kv_rgb = nn.Linear(dim, dim * 2, bias=False)
        self.kv_depth = nn.Linear(dim, dim * 2, bias=False)

    def forward(self, rgb_tokens, depth_tokens):
        batch_size, token_count, channels = rgb_tokens.shape

        q_rgb = rgb_tokens.reshape(batch_size, token_count, self.num_heads, channels // self.num_heads)
        q_rgb = q_rgb.permute(0, 2, 1, 3).contiguous()
        q_depth = depth_tokens.reshape(batch_size, token_count, self.num_heads, channels // self.num_heads)
        q_depth = q_depth.permute(0, 2, 1, 3).contiguous()

        k_rgb, v_rgb = self.kv_rgb(rgb_tokens).reshape(
            batch_size, token_count, 2, self.num_heads, channels // self.num_heads
        ).permute(2, 0, 3, 1, 4).contiguous()
        k_depth, v_depth = self.kv_depth(depth_tokens).reshape(
            batch_size, token_count, 2, self.num_heads, channels // self.num_heads
        ).permute(2, 0, 3, 1, 4).contiguous()

        ctx_rgb = (k_rgb.transpose(-2, -1) @ v_rgb) * self.scale
        ctx_rgb = ctx_rgb.softmax(dim=-2)
        ctx_depth = (k_depth.transpose(-2, -1) @ v_depth) * self.scale
        ctx_depth = ctx_depth.softmax(dim=-2)

        rgb_out = (q_rgb @ ctx_depth).permute(0, 2, 1, 3).reshape(batch_size, token_count, channels).contiguous()
        depth_out = (q_depth @ ctx_rgb).permute(0, 2, 1, 3).reshape(batch_size, token_count, channels).contiguous()
        return rgb_out, depth_out


class KTBCrossPath(nn.Module):
    """A small KTB-style cross path.

    We first project each modality into a smaller subspace, exchange context,
    then project back to the original width. This keeps the external shape
    fixed while still giving us a stronger mid/high-level interaction block.
    """

    def __init__(self, dim, reduction=2, num_heads=4):
        super().__init__()
        reduced_dim = dim // reduction
        if reduced_dim % num_heads != 0:
            raise ValueError(f"dim//reduction={reduced_dim} must be divisible by heads={num_heads}.")

        self.rgb_proj = nn.Linear(dim, reduced_dim * 2)
        self.depth_proj = nn.Linear(dim, reduced_dim * 2)
        self.rgb_act = nn.ReLU(inplace=True)
        self.depth_act = nn.ReLU(inplace=True)
        self.cross_attn = KTBCrossAttention(reduced_dim, num_heads=num_heads)
        self.rgb_out = nn.Linear(reduced_dim * 2, dim)
        self.depth_out = nn.Linear(reduced_dim * 2, dim)
        self.rgb_norm = nn.LayerNorm(dim)
        self.depth_norm = nn.LayerNorm(dim)

    def forward(self, rgb_tokens, depth_tokens):
        rgb_local, rgb_cross_seed = self.rgb_act(self.rgb_proj(rgb_tokens)).chunk(2, dim=-1)
        depth_local, depth_cross_seed = self.depth_act(self.depth_proj(depth_tokens)).chunk(2, dim=-1)

        rgb_cross, depth_cross = self.cross_attn(rgb_cross_seed, depth_cross_seed)

        rgb_mix = torch.cat([rgb_local, rgb_cross], dim=-1)
        depth_mix = torch.cat([depth_local, depth_cross], dim=-1)

        rgb_out = self.rgb_norm(rgb_tokens + self.rgb_out(rgb_mix))
        depth_out = self.depth_norm(depth_tokens + self.depth_out(depth_mix))
        return rgb_out, depth_out


class KTBChannelEmbed(nn.Module):
    """Project concatenated RGB/depth tokens back into one NCHW fused map."""

    def __init__(self, in_channels, out_channels, reduction=4):
        super().__init__()
        hidden_dim = max(out_channels // reduction, 1)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=True),
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                groups=hidden_dim,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(out_channels),
        )
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, tokens, height, width):
        batch_size, token_count, channels = tokens.shape
        feat = tokens.permute(0, 2, 1).reshape(batch_size, channels, height, width).contiguous()
        residual = self.residual(feat)
        fused = self.channel_embed(feat)
        return self.norm(residual + fused)


class KTBStyleFusionBlock(nn.Module):
    """KTB-inspired cross-modal fusion block for one feature stage.

    Input:
        rgb_feat   : (B, C, H, W)
        depth_feat : (B, C, H, W)
    Output:
        fused_feat : (B, C, H, W)

    Shape stays unchanged on purpose so the existing FPN decoder can stay
    untouched. This block is only dropped into c3/c4 in the current project.
    """

    def __init__(self, dim, reduction=4, token_reduction=2, num_heads=4):
        super().__init__()
        self.rectify = KTBFeatureRectify(dim, reduction=reduction)
        self.cross_path = KTBCrossPath(dim, reduction=token_reduction, num_heads=num_heads)
        self.channel_embed = KTBChannelEmbed(in_channels=dim * 2, out_channels=dim, reduction=reduction)
        self.refine = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb_feat, depth_feat):
        if rgb_feat.shape != depth_feat.shape:
            raise ValueError(
                f"KTBStyleFusionBlock expects same shape, got {tuple(rgb_feat.shape)} vs {tuple(depth_feat.shape)}."
            )

        rgb_rect, depth_rect = self.rectify(rgb_feat, depth_feat)
        batch_size, channels, height, width = rgb_rect.shape

        rgb_tokens = rgb_rect.flatten(2).transpose(1, 2)
        depth_tokens = depth_rect.flatten(2).transpose(1, 2)
        rgb_tokens, depth_tokens = self.cross_path(rgb_tokens, depth_tokens)

        merged_tokens = torch.cat([rgb_tokens, depth_tokens], dim=-1)
        fused = self.channel_embed(merged_tokens, height, width)

        # Residual preserve is important here: we want a stronger semantic fusion
        # on c3/c4, but we do not want to break the downstream FPN assumptions.
        fused = fused + 0.5 * (rgb_feat + depth_feat)
        return self.refine(fused)


class MidFusionSegmentor(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()
        self.rgb_encoder = RGBEncoder()
        self.depth_encoder = DepthEncoder()

        # c1/c2 keep the old fusion block for stability.
        self.low_level_fusions = nn.ModuleList(
            [
                GatedFusion(rgb_ch, depth_ch)
                for rgb_ch, depth_ch in zip(
                    self.rgb_encoder.out_channels[:2],
                    self.depth_encoder.out_channels[:2],
                )
            ]
        )

        # c3/c4 switch to the new KTB-style block.
        self.high_level_fusions = nn.ModuleList(
            [
                KTBStyleFusionBlock(self.rgb_encoder.out_channels[2], reduction=4, token_reduction=2, num_heads=4),
                KTBStyleFusionBlock(self.rgb_encoder.out_channels[3], reduction=4, token_reduction=4, num_heads=8),
            ]
        )

        self.decoder = SimpleFPNDecoder(self.rgb_encoder.out_channels, num_classes=num_classes)

    def forward(self, rgb, depth):
        rgb_feats = self.rgb_encoder(rgb)
        depth_feats = self.depth_encoder(depth)

        aligned_depth = []
        for rgb_feat, depth_feat in zip(rgb_feats, depth_feats):
            if rgb_feat.shape[-2:] != depth_feat.shape[-2:]:
                depth_feat = F.interpolate(depth_feat, size=rgb_feat.shape[-2:], mode="bilinear", align_corners=False)
            aligned_depth.append(depth_feat)

        fused_feats = []

        # stage c1/c2
        for idx in range(2):
            fused_feats.append(self.low_level_fusions[idx](rgb_feats[idx], aligned_depth[idx]))

        # stage c3/c4
        for idx in range(2, 4):
            fused_feats.append(self.high_level_fusions[idx - 2](rgb_feats[idx], aligned_depth[idx]))

        return self.decoder(fused_feats, input_size=rgb.shape[-2:])


class LitMidFusion(BaseLitSeg):
    def __init__(self, num_classes=40, lr=1e-4):
        super().__init__(num_classes=num_classes, lr=lr)
        self.model = MidFusionSegmentor(num_classes=num_classes)
