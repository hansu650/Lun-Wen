"""
Archived DFormer-guided depth fusion blocks.

These modules were tested in previous ablation experiments but are no longer part of the main training path.
They are kept only for reproducibility and documentation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_lit import BaseLitSeg
from src.models.decoder import SimpleFPNDecoder
from src.models.dformerv2_encoder import DFormerv2_S, load_dformerv2_pretrained
from src.models.encoder import DepthEncoder


class DFormerGuidedDepthAdapter(nn.Module):
    def __init__(self, primary_channels, depth_channels):
        super().__init__()
        hidden = max(primary_channels // 8, 16)
        relation_channels = primary_channels * 3

        self.depth_proj = nn.Sequential(
            nn.Conv2d(depth_channels, primary_channels, 1, bias=False),
            nn.BatchNorm2d(primary_channels),
        )
        self.guide_gate = nn.Sequential(
            nn.Conv2d(relation_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, primary_channels, 1),
            nn.Sigmoid(),
        )
        self.guide_update = nn.Sequential(
            nn.Conv2d(relation_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, primary_channels, 1, bias=False),
            nn.BatchNorm2d(primary_channels),
        )
        self.beta = nn.Parameter(torch.full((1, primary_channels, 1, 1), 1e-3))

    def forward(self, primary_feat, depth_feat):
        if depth_feat.shape[-2:] != primary_feat.shape[-2:]:
            depth_feat = F.interpolate(depth_feat, size=primary_feat.shape[-2:], mode="bilinear", align_corners=False)
        d = self.depth_proj(depth_feat)
        diff = torch.abs(d - primary_feat)
        relation = torch.cat([d, primary_feat, diff], dim=1)
        gate = self.guide_gate(relation)
        update = self.guide_update(relation)
        return d + self.beta * gate * update


class GuidedDepthEncoder(nn.Module):
    def __init__(self, primary_channels):
        super().__init__()
        self.base_depth_encoder = DepthEncoder()
        self.adapters = nn.ModuleList([
            DFormerGuidedDepthAdapter(primary_ch, depth_ch)
            for primary_ch, depth_ch in zip(primary_channels, self.base_depth_encoder.out_channels)
        ])
        self.out_channels = list(primary_channels)

    def forward(self, depth, dformer_feats):
        depth_feats = self.base_depth_encoder(depth)
        guided_depth_feats = []
        for adapter, primary_feat, depth_feat in zip(self.adapters, dformer_feats, depth_feats):
            guided_depth_feats.append(adapter(primary_feat, depth_feat))
        return guided_depth_feats


class AsymmetricComplementaryRectification(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 8, 16)
        relation_channels = channels * 3

        self.depth_rect_gate = nn.Sequential(
            nn.Conv2d(relation_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.depth_rect_update = nn.Sequential(
            nn.Conv2d(relation_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.primary_rect_gate = nn.Sequential(
            nn.Conv2d(relation_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.primary_rect_update = nn.Sequential(
            nn.Conv2d(relation_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.alpha_d = nn.Parameter(torch.full((1, channels, 1, 1), 1e-3))
        self.alpha_p = nn.Parameter(torch.full((1, channels, 1, 1), 1e-4))

    def forward(self, primary_feat, guided_depth_feat):
        diff = torch.abs(guided_depth_feat - primary_feat)
        x = torch.cat([guided_depth_feat, primary_feat, diff], dim=1)
        d_rect = guided_depth_feat + self.alpha_d * self.depth_rect_gate(x) * self.depth_rect_update(x)
        p_rect = primary_feat + self.alpha_p * self.primary_rect_gate(x) * self.primary_rect_update(x)
        return p_rect, d_rect


class AttentionComplementaryAggregation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 8, 16)
        attn_hidden = max(channels // 16, 16)
        relation_channels = channels * 3

        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(relation_channels, attn_hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(attn_hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.comp_adapter = nn.Sequential(
            nn.Conv2d(relation_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1), 1e-3))

    def forward(self, primary_feat, rectified_depth_feat):
        diff = torch.abs(primary_feat - rectified_depth_feat)
        attn_input = torch.cat([primary_feat, rectified_depth_feat, diff], dim=1)
        channel = self.channel_attn(attn_input)
        spatial_input = torch.cat([
            torch.mean(attn_input, dim=1, keepdim=True),
            torch.max(attn_input, dim=1, keepdim=True)[0],
        ], dim=1)
        spatial = self.spatial_attn(spatial_input)
        comp_input = torch.cat([primary_feat, channel * rectified_depth_feat, diff], dim=1)
        comp = self.comp_adapter(comp_input)
        return primary_feat + self.gamma * spatial * comp


class DFormerGuidedDepthCompFusionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.rectification = AsymmetricComplementaryRectification(channels)
        self.aggregation = AttentionComplementaryAggregation(channels)

    def forward(self, primary_feat, guided_depth_feat):
        p_rect, d_rect = self.rectification(primary_feat, guided_depth_feat)
        out_from_aggregation = self.aggregation(p_rect, d_rect)
        return primary_feat + (out_from_aggregation - p_rect)


class DFormerV2GuidedDepthCompFusionSegmentor(nn.Module):
    def __init__(self, num_classes=40, dformerv2_pretrained=None):
        super().__init__()
        self.rgb_encoder = DFormerv2_S()
        if dformerv2_pretrained:
            self.pretrained_load_stats = load_dformerv2_pretrained(self.rgb_encoder, dformerv2_pretrained)
        else:
            self.pretrained_load_stats = None
        self.guided_depth_encoder = GuidedDepthEncoder(
            primary_channels=self.rgb_encoder.out_channels,
        )
        self.fusions = nn.ModuleList([
            DFormerGuidedDepthCompFusionBlock(ch)
            for ch in self.rgb_encoder.out_channels
        ])
        self.decoder = SimpleFPNDecoder(self.rgb_encoder.out_channels, num_classes=num_classes)

    def forward(self, rgb, depth):
        dformer_feats = self.rgb_encoder(rgb, depth)
        guided_depth_feats = self.guided_depth_encoder(depth, dformer_feats)

        fused_feats = []
        for fusion, primary_feat, guided_depth_feat in zip(self.fusions, dformer_feats, guided_depth_feats):
            if guided_depth_feat.shape[-2:] != primary_feat.shape[-2:]:
                guided_depth_feat = F.interpolate(
                    guided_depth_feat,
                    size=primary_feat.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            fused_feats.append(fusion(primary_feat, guided_depth_feat))
        return self.decoder(fused_feats, input_size=rgb.shape[-2:])


class LitDFormerV2GuidedDepthCompFusion(BaseLitSeg):
    def __init__(self, num_classes=40, lr=1e-4, dformerv2_pretrained=None):
        super().__init__(num_classes=num_classes, lr=lr)
        self.model = DFormerV2GuidedDepthCompFusionSegmentor(
            num_classes=num_classes,
            dformerv2_pretrained=dformerv2_pretrained,
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)


class DFormerGuidedDepthAdapterSimpleFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 8, 16)
        self.residual_adapter = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1), 1e-3))

    def forward(self, primary_feat, guided_depth_feat):
        if guided_depth_feat.shape[-2:] != primary_feat.shape[-2:]:
            guided_depth_feat = F.interpolate(
                guided_depth_feat,
                size=primary_feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        diff = torch.abs(primary_feat - guided_depth_feat)
        x = torch.cat([guided_depth_feat, diff], dim=1)
        delta = self.residual_adapter(x)
        return primary_feat + self.gamma * delta


class DFormerV2GuidedDepthAdapterSimpleSegmentor(nn.Module):
    def __init__(self, num_classes=40, dformerv2_pretrained=None):
        super().__init__()
        self.rgb_encoder = DFormerv2_S()
        if dformerv2_pretrained:
            self.pretrained_load_stats = load_dformerv2_pretrained(self.rgb_encoder, dformerv2_pretrained)
        else:
            self.pretrained_load_stats = None
        self.guided_depth_encoder = GuidedDepthEncoder(
            primary_channels=self.rgb_encoder.out_channels,
        )
        self.fusions = nn.ModuleList([
            DFormerGuidedDepthAdapterSimpleFusion(ch)
            for ch in self.rgb_encoder.out_channels
        ])
        self.decoder = SimpleFPNDecoder(self.rgb_encoder.out_channels, num_classes=num_classes)

    def forward(self, rgb, depth):
        dformer_feats = self.rgb_encoder(rgb, depth)
        guided_depth_feats = self.guided_depth_encoder(depth, dformer_feats)

        fused_feats = []
        for fusion, primary_feat, guided_depth_feat in zip(self.fusions, dformer_feats, guided_depth_feats):
            if guided_depth_feat.shape[-2:] != primary_feat.shape[-2:]:
                guided_depth_feat = F.interpolate(
                    guided_depth_feat,
                    size=primary_feat.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            fused_feats.append(fusion(primary_feat, guided_depth_feat))
        return self.decoder(fused_feats, input_size=rgb.shape[-2:])


class LitDFormerV2GuidedDepthAdapterSimple(BaseLitSeg):
    def __init__(self, num_classes=40, lr=1e-4, dformerv2_pretrained=None):
        super().__init__(num_classes=num_classes, lr=lr)
        self.model = DFormerV2GuidedDepthAdapterSimpleSegmentor(
            num_classes=num_classes,
            dformerv2_pretrained=dformerv2_pretrained,
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
