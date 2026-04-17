"""Mid-fusion model with KTB-style fusion on c3/c4 and FDAM-style enhancement after fusion."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_lit import BaseLitSeg
from .decoder import SimpleFPNDecoder
from .encoder import DepthEncoder, RGBEncoder


def nchw_to_nlc(x: torch.Tensor) -> torch.Tensor:
    """Flatten (B, C, H, W) into token layout (B, H*W, C)."""

    return x.flatten(2).transpose(1, 2).contiguous()


def nlc_to_nchw(x: torch.Tensor, hw_shape: tuple[int, int]) -> torch.Tensor:
    """Restore token layout (B, H*W, C) back to (B, C, H, W)."""

    height, width = hw_shape
    batch_size, token_count, channels = x.shape
    if token_count != height * width:
        raise ValueError(f"Token count {token_count} does not match target size {height}x{width}.")
    return x.transpose(1, 2).reshape(batch_size, channels, height, width).contiguous()


class GatedFusion(nn.Module):
    """Simple low-level fusion kept from the original project.

    c1/c2 still carry a lot of edges and local texture. Keeping the old gate on
    these levels helps us isolate the effect of the new higher-level modules.
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
    """KTB FRM-style channel reweighting."""

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
    """KTB FRM idea: let RGB/depth first rectify each other before fusion."""

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
    """Simplified KTB-style token cross attention."""

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
    """KTB CrossPath idea: project down, exchange context, project back."""

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
    """Project concatenated RGB/depth tokens back to one NCHW fused map."""

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
        batch_size, _, channels = tokens.shape
        feat = tokens.permute(0, 2, 1).reshape(batch_size, channels, height, width).contiguous()
        residual = self.residual(feat)
        fused = self.channel_embed(feat)
        return self.norm(residual + fused)


class KTBStyleFusionBlock(nn.Module):
    """Block 1: KTB-inspired fusion block.

    Input:
        rgb_feat   : (B, C, H, W)
        depth_feat : (B, C, H, W)
    Output:
        fused_feat : (B, C, H, W)

    Shape is preserved so the downstream FPN decoder does not need any change.
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
        _, _, height, width = rgb_rect.shape

        rgb_tokens = nchw_to_nlc(rgb_rect)
        depth_tokens = nchw_to_nlc(depth_rect)
        rgb_tokens, depth_tokens = self.cross_path(rgb_tokens, depth_tokens)

        merged_tokens = torch.cat([rgb_tokens, depth_tokens], dim=-1)
        fused = self.channel_embed(merged_tokens, height, width)

        # Preserve original semantics so the new block behaves as a plug-and-play
        # upgrade instead of fully rewriting the feature distribution.
        fused = fused + 0.5 * (rgb_feat + depth_feat)
        return self.refine(fused)


class StarReLU(nn.Module):
    """FDAM uses StarReLU before predicting dynamic frequency weights."""

    def __init__(self, scale_value=1.0, bias_value=0.0):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.scale = nn.Parameter(scale_value * torch.ones(1))
        self.bias = nn.Parameter(bias_value * torch.ones(1))

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias


class FdamTokenMlp(nn.Module):
    """Light token MLP used inside the FDAM-style enhancement block."""

    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class AttentionwithAttInvLite(nn.Module):
    """Simplified FDAM attention.

    Keep the core idea:
    - standard multi-head self-attention
    - a low-frequency modulation path on the attention output
    - a complementary high-frequency path using (value - attention_output)
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by heads {num_heads}.")

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pre_act = StarReLU()
        self.low_freq_predictor = nn.Linear(dim, num_heads, bias=True)
        self.high_freq_predictor = nn.Linear(dim, num_heads, bias=True)
        self.lf_gamma = nn.Parameter(1e-5 * torch.ones(dim))
        self.hf_gamma = nn.Parameter(1e-5 * torch.ones(dim))

    def forward(self, x):
        batch_size, token_count, channels = x.shape
        detail_feat = self.pre_act(x)

        low_freq = self.low_freq_predictor(detail_feat).tanh_()
        low_freq = low_freq.reshape(batch_size, token_count, self.num_heads, 1).repeat(
            1, 1, 1, channels // self.num_heads
        )
        low_freq = low_freq.reshape(batch_size, token_count, channels)

        high_freq = F.softplus(self.high_freq_predictor(detail_feat))
        high_freq_sq = high_freq ** 2
        high_freq = 2.0 * high_freq_sq / (high_freq_sq + 0.3678)
        high_freq = high_freq.reshape(batch_size, token_count, self.num_heads, 1).repeat(
            1, 1, 1, channels // self.num_heads
        )
        high_freq = high_freq.reshape(batch_size, token_count, channels)

        qkv = self.qkv(x).reshape(
            batch_size, token_count, 3, self.num_heads, channels // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn_out = (attn @ v).transpose(1, 2).reshape(batch_size, token_count, channels)

        value_tokens = v.permute(0, 2, 1, 3).reshape(batch_size, token_count, channels)
        high_freq_residual = value_tokens - attn_out

        attn_out = attn_out + attn_out * low_freq * self.lf_gamma.view(1, 1, -1)
        attn_out = attn_out + high_freq * high_freq_residual * self.hf_gamma.view(1, 1, -1)

        attn_out = self.proj(attn_out)
        attn_out = self.proj_drop(attn_out)
        return attn_out


class GroupDynamicScaleLite(nn.Module):
    """Simplified FDAM frequency dynamic scaling.

    Input / output are both (B, C, H, W). We go to frequency space with rFFT,
    predict group-wise filter routing from the spatial mean, then scale the
    frequency coefficients and transform back. This keeps shape unchanged.
    """

    def __init__(
        self,
        dim,
        expansion_ratio=0.125,
        num_filters=4,
        base_size=14,
        group=16,
        init_scale=1e-5,
    ):
        super().__init__()
        if dim % group != 0:
            raise ValueError(f"dim {dim} must be divisible by group {group}.")

        self.dim = dim
        self.group = group
        self.num_filters = num_filters
        self.base_size = base_size
        self.base_filter_size = base_size // 2 + 1

        hidden_dim = max(int(dim * expansion_ratio), 1)
        self.reweight = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=False),
            StarReLU(),
            nn.Linear(hidden_dim, group * num_filters, bias=False),
        )

        # FDAM uses learnable frequency filters that are later resized to the
        # actual rFFT size. We keep that idea but make the implementation small.
        self.frequency_weights = nn.Parameter(
            torch.randn(num_filters, dim // group, self.base_size, self.base_filter_size) * init_scale
        )

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        input_dtype = x.dtype

        # rFFT is easier to keep numerically stable in float32.
        x_freq = torch.fft.rfft2(x.to(torch.float32), dim=(2, 3), norm="ortho")
        _, _, freq_h, freq_w = x_freq.shape

        routing = self.reweight(x.mean(dim=(2, 3))).view(batch_size, self.group, self.num_filters).tanh_()

        weight_bank = self.frequency_weights
        if weight_bank.shape[-2:] != (freq_h, freq_w):
            weight_bank = F.interpolate(weight_bank, size=(freq_h, freq_w), mode="bicubic", align_corners=True)

        dynamic_weights = torch.einsum("bgf,fchw->bgchw", routing, weight_bank).reshape(
            batch_size, channels, freq_h, freq_w
        )

        x_freq = torch.view_as_complex(
            torch.stack(
                [
                    x_freq.real * dynamic_weights,
                    x_freq.imag * dynamic_weights,
                ],
                dim=-1,
            )
        )
        x = torch.fft.irfft2(x_freq, s=(height, width), dim=(2, 3), norm="ortho")
        return x.to(input_dtype)


class FdamLocalFreqEnhanceBlock(nn.Module):
    """Block 2: FDAM-style local/frequency enhancement.

    Input:
        fused_feat: (B, C, H, W)
    Output:
        enhanced_feat: (B, C, H, W)

    We keep the main FDAM recipe:
    1. replace plain attention with AttentionwithAttInvLite
    2. add GroupDynamicScaleLite after attention
    3. add GroupDynamicScaleLite after MLP
    """

    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, group=16, base_size=14):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = AttentionwithAttInvLite(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FdamTokenMlp(dim, mlp_ratio=mlp_ratio)

        self.gamma_1 = nn.Parameter(1e-4 * torch.ones(dim))
        self.gamma_2 = nn.Parameter(1e-4 * torch.ones(dim))

        self.freq_scale_1 = GroupDynamicScaleLite(dim=dim, group=group, num_filters=4, base_size=base_size)
        self.freq_scale_2 = GroupDynamicScaleLite(dim=dim, group=group, num_filters=4, base_size=base_size)

    def forward(self, fused_feat):
        batch_size, channels, height, width = fused_feat.shape
        tokens = nchw_to_nlc(fused_feat)

        att_tokens = self.attn(self.norm1(tokens))
        att_feat = nlc_to_nchw(att_tokens, (height, width))
        att_feat = self.freq_scale_1(att_feat) + att_feat
        att_tokens = nchw_to_nlc(att_feat)
        tokens = tokens + self.gamma_1.view(1, 1, -1) * att_tokens

        mlp_tokens = self.mlp(self.norm2(tokens))
        mlp_feat = nlc_to_nchw(mlp_tokens, (height, width))
        mlp_feat = self.freq_scale_2(mlp_feat) + mlp_feat
        mlp_tokens = nchw_to_nlc(mlp_feat)
        tokens = tokens + self.gamma_2.view(1, 1, -1) * mlp_tokens

        enhanced_feat = nlc_to_nchw(tokens, (height, width))
        return enhanced_feat


class MidFusionSegmentor(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()
        self.rgb_encoder = RGBEncoder()
        self.depth_encoder = DepthEncoder()

        # c1/c2 still use the original lightweight gate.
        self.low_level_fusions = nn.ModuleList(
            [
                GatedFusion(rgb_ch, depth_ch)
                for rgb_ch, depth_ch in zip(
                    self.rgb_encoder.out_channels[:2],
                    self.depth_encoder.out_channels[:2],
                )
            ]
        )

        # c3/c4 use block 1: KTB-style cross-modal fusion.
        self.high_level_fusions = nn.ModuleList(
            [
                KTBStyleFusionBlock(self.rgb_encoder.out_channels[2], reduction=4, token_reduction=2, num_heads=4),
                KTBStyleFusionBlock(self.rgb_encoder.out_channels[3], reduction=4, token_reduction=4, num_heads=8),
            ]
        )

        # c3/c4 then pass through block 2: FDAM-style local/frequency enhancement.
        self.high_level_enhancers = nn.ModuleList(
            [
                FdamLocalFreqEnhanceBlock(self.rgb_encoder.out_channels[2], num_heads=8, mlp_ratio=4.0, group=16, base_size=14),
                FdamLocalFreqEnhanceBlock(self.rgb_encoder.out_channels[3], num_heads=8, mlp_ratio=4.0, group=32, base_size=14),
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

        # c1/c2: old gate only.
        for idx in range(2):
            fused_feats.append(self.low_level_fusions[idx](rgb_feats[idx], aligned_depth[idx]))

        # c3/c4: KTB fusion first, then FDAM enhancement.
        for idx in range(2, 4):
            fused = self.high_level_fusions[idx - 2](rgb_feats[idx], aligned_depth[idx])
            fused = self.high_level_enhancers[idx - 2](fused)
            fused_feats.append(fused)

        return self.decoder(fused_feats, input_size=rgb.shape[-2:])


class LitMidFusion(BaseLitSeg):
    def __init__(self, num_classes=40, lr=1e-4):
        super().__init__(num_classes=num_classes, lr=lr)
        self.model = MidFusionSegmentor(num_classes=num_classes)
