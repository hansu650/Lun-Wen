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


class DepthAwareLocalRefineFusion(nn.Module):
    """Clean ablation c3 fusion block.

    Goal:
    - keep the stable gate-style fused base
    - let depth act as a light geometry prior
    - add only a small local refine branch

    Input:
        rgb_feat   : (B, C, H, W)
        depth_feat : (B, C_d, H, W)
    Output:
        fused_feat : (B, C, H, W)

    The output shape is unchanged so the downstream decoder/FPN does not need
    any modification.
    """

    def __init__(self, rgb_channels, depth_channels, reduction=4):
        super().__init__()
        self.depth_proj = nn.Conv2d(depth_channels, rgb_channels, kernel_size=1, bias=False)

        # Keep a simple base fusion first so this block is still anchored to the
        # project's original stable fusion path.
        self.base_gate = nn.Sequential(
            nn.Conv2d(rgb_channels * 2, rgb_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(rgb_channels),
            nn.Sigmoid(),
        )

        hidden_dim = max(rgb_channels // reduction, 1)

        # Channel guidance from depth: (B, C, H, W) -> (B, C, 1, 1)
        self.channel_guidance = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(rgb_channels, hidden_dim, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, rgb_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # Spatial guidance from depth: (B, C, H, W) -> (B, 1, H, W)
        self.spatial_guidance = nn.Sequential(
            nn.Conv2d(rgb_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )

        # Keep the local branch lightweight and stable. This earlier version was
        # the strongest clean c3 design before the later experimental upgrades.
        self.local_refine = nn.Sequential(
            nn.Conv2d(rgb_channels, rgb_channels, kernel_size=3, padding=1, groups=rgb_channels, bias=False),
            nn.BatchNorm2d(rgb_channels),
            nn.GELU(),
            nn.Conv2d(rgb_channels, rgb_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(rgb_channels),
        )

        self.out_norm = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(rgb_channels, rgb_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(rgb_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb_feat, depth_feat):
        depth_feat = self.depth_proj(depth_feat)

        # Step 1: simple stable base fusion
        gate = self.base_gate(torch.cat([rgb_feat, depth_feat], dim=1))
        base_fused = gate * rgb_feat + (1.0 - gate) * depth_feat

        # Step 2: depth-guided modulation
        channel_guide = self.channel_guidance(depth_feat)    # (B, C, 1, 1)
        spatial_guide = self.spatial_guidance(depth_feat)    # (B, 1, H, W)

        guided = base_fused * (1.0 + channel_guide)
        guided = guided * (1.0 + spatial_guide)

        # Step 3: lightweight local refinement with residual return
        refined = guided + self.local_refine(guided)
        return self.out_norm(base_fused + refined)


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


class DepthPromptGenerator(nn.Module):
    """Generate prompt tokens from c4 depth features for the geometry block.

    The prompt generator stays lightweight, while the attention mixer will add
    the DFormerv2-style geometry prior on top of these prompt/image tokens.
    """

    def __init__(self, dim, prompt_hw=(2, 4)):
        super().__init__()
        self.prompt_hw = prompt_hw
        self.num_prompts = prompt_hw[0] * prompt_hw[1]

        self.depth_prior = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.prompt_pool = nn.AdaptiveAvgPool2d(prompt_hw)
        self.prompt_embed = nn.Parameter(torch.zeros(1, self.num_prompts, dim))
        nn.init.trunc_normal_(self.prompt_embed, std=0.02)
        self.norm = nn.LayerNorm(dim)

    def forward(self, depth_feat):
        batch_size = depth_feat.shape[0]
        depth_prior = self.depth_prior(depth_feat)
        prompt_map = self.prompt_pool(depth_prior)
        prompt_tokens = nchw_to_nlc(prompt_map)
        prompt_tokens = prompt_tokens + self.prompt_embed.expand(batch_size, -1, -1)
        return self.norm(prompt_tokens)


class PromptTokenMixer(nn.Module):
    """Original lightweight prompt mixer used in the stable baseline.

    The design is intentionally small:
    - one pre-norm multi-head self-attention
    - one small MLP
    - residual connections
    """

    def __init__(self, dim, num_heads=8, mlp_ratio=2.0, drop=0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, joint_tokens):
        attn_input = self.norm1(joint_tokens)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        joint_tokens = joint_tokens + attn_output
        joint_tokens = joint_tokens + self.mlp(self.norm2(joint_tokens))
        return joint_tokens


class DepthPromptTokenBlock(nn.Module):
    """Stable c4 prompt block baseline.

    Input:
        fused_feat_c4 : (B, C, H, W)
        depth_feat_c4 : (B, C, H, W)
    Output:
        enhanced_feat : (B, C, H, W)
    """

    def __init__(self, dim, prompt_hw=(2, 4), num_heads=8, mlp_ratio=2.0, drop=0.0):
        super().__init__()
        self.prompt_generator = DepthPromptGenerator(dim=dim, prompt_hw=prompt_hw)
        self.token_mixer = PromptTokenMixer(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop)
        self.out_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )

    def forward(self, fused_feat_c4, depth_feat_c4):
        if fused_feat_c4.shape != depth_feat_c4.shape:
            raise ValueError(
                "DepthPromptTokenBlock expects fused c4 and depth c4 to share the same shape, "
                f"got {tuple(fused_feat_c4.shape)} vs {tuple(depth_feat_c4.shape)}."
            )

        _, _, height, width = fused_feat_c4.shape

        image_tokens = nchw_to_nlc(fused_feat_c4)                 # (B, H*W, C)
        prompt_tokens = self.prompt_generator(depth_feat_c4)      # (B, K,   C)
        num_image_tokens = image_tokens.shape[1]

        # Stable baseline design: append prompt tokens after image tokens,
        # mix once, then discard prompts and keep refined image tokens only.
        joint_tokens = torch.cat([image_tokens, prompt_tokens], dim=1)
        joint_tokens = self.token_mixer(joint_tokens)
        image_tokens = joint_tokens[:, :num_image_tokens, :]
        enhanced_feat = nlc_to_nchw(image_tokens, (height, width))

        return fused_feat_c4 + self.out_proj(enhanced_feat)


class GeometryPriorGenerator(nn.Module):
    """Lightweight DFormerv2-style geometry bias generator.

    Adapted from DFormerv2's GeoPriorGen, generate_pos_decay and
    generate_depth_decay. We keep the core idea:
    - a positional decay prior based on token coordinates
    - a depth decay prior based on depth-token differences
    - a learned blend between the two
    """

    def __init__(self, dim, num_heads, prompt_hw=(2, 4), initial_value=1.0, heads_range=4.0):
        super().__init__()
        self.num_heads = num_heads
        self.prompt_hw = prompt_hw
        self.num_prompts = prompt_hw[0] * prompt_hw[1]

        decay = torch.log(
            1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads)
        )
        self.register_buffer("decay", decay)
        self.weight = nn.Parameter(torch.ones(2, 1, 1, 1), requires_grad=True)

        self.depth_scalar = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, 1, kernel_size=1, bias=True),
        )

    def _build_prompt_coords(self, device, dtype):
        ph, pw = self.prompt_hw
        y = torch.linspace(0, 1, ph, device=device, dtype=dtype)
        x = torch.linspace(0, 1, pw, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([grid_y, grid_x], dim=-1).reshape(-1, 2)

    def _build_image_coords(self, height, width, device, dtype):
        y = torch.linspace(0, 1, height, device=device, dtype=dtype)
        x = torch.linspace(0, 1, width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([grid_y, grid_x], dim=-1).reshape(-1, 2)

    def forward(self, depth_feat):
        batch_size, _, height, width = depth_feat.shape
        device = depth_feat.device
        dtype = depth_feat.dtype

        depth_scalar_map = self.depth_scalar(depth_feat)                 # (B, 1, H, W)
        image_depth = depth_scalar_map.flatten(2).transpose(1, 2)        # (B, N, 1)
        prompt_depth = F.adaptive_avg_pool2d(depth_scalar_map, self.prompt_hw)
        prompt_depth = prompt_depth.flatten(2).transpose(1, 2)           # (B, K, 1)
        joint_depth = torch.cat([prompt_depth, image_depth], dim=1)      # (B, K+N, 1)

        prompt_coords = self._build_prompt_coords(device, dtype)
        image_coords = self._build_image_coords(height, width, device, dtype)
        joint_coords = torch.cat([prompt_coords, image_coords], dim=0)    # (K+N, 2)
        pos_delta = joint_coords[:, None, :] - joint_coords[None, :, :]
        pos_mask = pos_delta.abs().sum(dim=-1)                            # (K+N, K+N)
        pos_mask = pos_mask.unsqueeze(0) * self.decay[:, None, None]      # (heads, K+N, K+N)

        depth_delta = joint_depth[:, :, None, :] - joint_depth[:, None, :, :]
        depth_mask = depth_delta.abs().sum(dim=-1)                        # (B, K+N, K+N)
        depth_mask = depth_mask.unsqueeze(1) * self.decay[None, :, None, None]

        return self.weight[0] * pos_mask.unsqueeze(0) + self.weight[1] * depth_mask


class GeometryAwarePromptMixer(nn.Module):
    """Prompt-aware attention with DFormerv2-style geometry bias on logits."""

    def __init__(self, dim, num_heads=8, mlp_ratio=2.0, drop=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        hidden_dim = int(dim * mlp_ratio)

        self.norm_qkv = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(drop)

        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, prompt_tokens, image_tokens, geometry_bias):
        joint_tokens = torch.cat([prompt_tokens, image_tokens], dim=1)    # (B, K+N, C)
        residual = joint_tokens

        x = self.norm_qkv(joint_tokens)
        batch_size, token_count, channels = x.shape

        q = self.q_proj(x).reshape(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)

        attn_logits = (q @ k.transpose(-2, -1)) * self.scale
        attn_logits = attn_logits + geometry_bias
        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.attn_drop(attn)

        attn_out = attn @ v
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, token_count, channels)
        joint_tokens = residual + self.out_proj(attn_out)
        joint_tokens = joint_tokens + self.mlp(self.norm_mlp(joint_tokens))

        num_prompts = prompt_tokens.shape[1]
        return joint_tokens[:, num_prompts:, :]


class GeometryAwarePromptTokenBlock(nn.Module):
    """DFormerv2-inspired c4 prompt block with geometry-aware attention bias."""

    def __init__(self, dim, prompt_hw=(2, 4), num_heads=8, mlp_ratio=2.0, drop=0.0):
        super().__init__()
        self.prompt_generator = DepthPromptGenerator(dim=dim, prompt_hw=prompt_hw)
        self.geometry_prior = GeometryPriorGenerator(dim=dim, num_heads=num_heads, prompt_hw=prompt_hw)
        self.token_mixer = GeometryAwarePromptMixer(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=drop,
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )

    def forward(self, fused_feat_c4, depth_feat_c4):
        if fused_feat_c4.shape != depth_feat_c4.shape:
            raise ValueError(
                "GeometryAwarePromptTokenBlock expects fused c4 and depth c4 to share the same shape, "
                f"got {tuple(fused_feat_c4.shape)} vs {tuple(depth_feat_c4.shape)}."
            )

        _, _, height, width = fused_feat_c4.shape

        image_tokens = nchw_to_nlc(fused_feat_c4)                    # (B, N, C)
        prompt_tokens = self.prompt_generator(depth_feat_c4)         # (B, K, C)
        geometry_bias = self.geometry_prior(depth_feat_c4)           # (B, heads, K+N, K+N)

        image_tokens = self.token_mixer(prompt_tokens, image_tokens, geometry_bias)
        enhanced_feat = nlc_to_nchw(image_tokens, (height, width))
        return fused_feat_c4 + self.out_proj(enhanced_feat)


class MidFusionSegmentor(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()
        # 当前实验只替换 backbone：
        # - RGBEncoder 在 encoder.py 里已经改成 DINOv2-S 路线
        # - DepthEncoder 在 encoder.py 里已经改成 Swin-T 路线
        # 下面的 fusion 和 decoder 先沿用原来的结构，方便观察 backbone 互换本身的影响。
        self.rgb_encoder = RGBEncoder()
        self.depth_encoder = DepthEncoder()

        # c1 keeps the original simple gate.
        self.c1_fusion = GatedFusion(
            self.rgb_encoder.out_channels[0],
            self.depth_encoder.out_channels[0],
        )

        # c2 returns to the original simple gated fusion for the stable
        # baseline.
        self.c2_fusion = GatedFusion(
            self.rgb_encoder.out_channels[1],
            self.depth_encoder.out_channels[1],
        )

        # c3 keeps the current stable depth-aware local refine fusion.
        self.c3_fusion = DepthAwareLocalRefineFusion(
            rgb_channels=self.rgb_encoder.out_channels[2],
            depth_channels=self.depth_encoder.out_channels[2],
            reduction=4,
        )

        # c4 uses a simple fusion first, then the original prompt-token block
        # that gave the strongest stable result.
        self.c4_fusion = GatedFusion(
            self.rgb_encoder.out_channels[3],
            self.depth_encoder.out_channels[3],
        )
        self.c4_prompt_block = DepthPromptTokenBlock(
            dim=self.rgb_encoder.out_channels[3],
            prompt_hw=(2, 4),
            num_heads=8,
            mlp_ratio=2.0,
            drop=0.0,
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

        fused_feats.append(self.c1_fusion(rgb_feats[0], aligned_depth[0]))
        fused_feats.append(self.c2_fusion(rgb_feats[1], aligned_depth[1]))

        fused_c3 = self.c3_fusion(rgb_feats[2], aligned_depth[2])
        fused_feats.append(fused_c3)

        fused_c4 = self.c4_fusion(rgb_feats[3], aligned_depth[3])
        fused_c4 = self.c4_prompt_block(fused_c4, aligned_depth[3])
        fused_feats.append(fused_c4)

        return self.decoder(fused_feats, input_size=rgb.shape[-2:])


class LitMidFusion(BaseLitSeg):
    def __init__(self, num_classes=40, lr=1e-4):
        super().__init__(num_classes=num_classes, lr=lr)
        self.model = MidFusionSegmentor(num_classes=num_classes)
