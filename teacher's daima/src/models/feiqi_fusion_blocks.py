"""
Archived experimental fusion blocks.

These modules were tested in ablation experiments but are no longer part of the main training path.
They are kept only for reproducibility and documentation.

Main reason:
DGC-AF++ was near baseline but did not provide stable improvement across repeated runs.
CSG and GRM-ARD were negative results.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import DepthEncoder
from .dformerv2_encoder import DFormerv2_S, load_dformerv2_pretrained
from .decoder import SimpleFPNDecoder
from .base_lit import BaseLitSeg


class DepthReliabilityHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 4, 16)
        self.spatial_reliability = nn.Sequential(
            nn.Conv2d(channels * 3, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.channel_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_reliability = nn.Sequential(
            nn.Conv2d(channels * 3, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, rel_input):
        spatial = self.spatial_reliability(rel_input)
        channel = self.channel_reliability(self.channel_pool(rel_input))
        return spatial, channel


class PrimaryGuidedSparseDepthCompensation(nn.Module):
    def __init__(self, rgb_channels, depth_channels):
        super().__init__()
        self.depth_proj = nn.Sequential(
            nn.Conv2d(depth_channels, rgb_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(rgb_channels),
        )
        self.reliability = DepthReliabilityHead(rgb_channels)
        self.residual_adapter = nn.Sequential(
            nn.Conv2d(rgb_channels * 2, rgb_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(rgb_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                rgb_channels,
                rgb_channels,
                kernel_size=3,
                padding=1,
                groups=rgb_channels,
                bias=False,
            ),
            nn.BatchNorm2d(rgb_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(rgb_channels, rgb_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(rgb_channels),
        )
        self.gamma = nn.Parameter(torch.full((1, rgb_channels, 1, 1), 1e-3))

    def forward(self, primary_feat, depth_feat):
        depth = self.depth_proj(depth_feat)
        diff = torch.abs(primary_feat - depth)
        rel_input = torch.cat([primary_feat, depth, diff], dim=1)
        spatial, channel = self.reliability(rel_input)
        adapter_input = torch.cat([depth, diff], dim=1)
        delta = self.residual_adapter(adapter_input)
        return primary_feat + self.gamma * spatial * channel * delta


class DFormerGuidedCyclicAttentionFusion(nn.Module):
    def __init__(self, rgb_channels, depth_channels):
        super().__init__()
        hidden = max(rgb_channels // 8, 16)
        self.depth_proj = nn.Sequential(
            nn.Conv2d(depth_channels, rgb_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(rgb_channels),
        )
        self.clean_gate = nn.Sequential(
            nn.Conv2d(rgb_channels * 3, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=3,
                padding=1,
                groups=hidden,
                bias=False,
            ),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, rgb_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.query = nn.Sequential(
            nn.Conv2d(rgb_channels, rgb_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(rgb_channels),
        )
        self.key = nn.Sequential(
            nn.Conv2d(rgb_channels, rgb_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(rgb_channels),
        )
        self.value = nn.Sequential(
            nn.Conv2d(rgb_channels, rgb_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(rgb_channels),
        )
        self.support_score = nn.Sequential(
            nn.Conv2d(
                rgb_channels,
                rgb_channels,
                kernel_size=3,
                padding=1,
                groups=rgb_channels,
                bias=False,
            ),
            nn.BatchNorm2d(rgb_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(rgb_channels, rgb_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.support = nn.Sequential(
            nn.Conv2d(rgb_channels * 3, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=3,
                padding=1,
                groups=hidden,
                bias=False,
            ),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, rgb_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.conflict = nn.Sequential(
            nn.Conv2d(rgb_channels * 3, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=3,
                padding=1,
                groups=hidden,
                bias=False,
            ),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, rgb_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.residual_adapter = nn.Sequential(
            nn.Conv2d(rgb_channels * 2, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=3,
                padding=1,
                groups=hidden,
                bias=False,
            ),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, rgb_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(rgb_channels),
        )
        self.beta = nn.Parameter(torch.full((1, rgb_channels, 1, 1), 1e-3))
        self.gamma = nn.Parameter(torch.full((1, rgb_channels, 1, 1), 1e-3))
        self.sparse_k = nn.Parameter(torch.full((1, rgb_channels, 1, 1), 5.0))
        self.sparse_tau = nn.Parameter(torch.zeros(1, rgb_channels, 1, 1))

    def forward(self, primary_feat, depth_feat):
        depth = self.depth_proj(depth_feat)
        diff0 = torch.abs(primary_feat - depth)
        clean_input = torch.cat([primary_feat, depth, diff0], dim=1)
        clean_gate = self.clean_gate(clean_input)
        depth_clean = depth * (1 + self.beta * (2 * clean_gate - 1))

        query = self.query(primary_feat)
        key = self.key(depth_clean)
        value = self.value(depth_clean)
        support_attn = self.support_score(query * key)
        depth_attn = support_attn * value

        support_input = torch.cat([primary_feat, depth_attn, primary_feat * depth_attn], dim=1)
        support = self.support(support_input)
        conflict_input = torch.cat([primary_feat, depth_attn, torch.abs(primary_feat - depth_attn)], dim=1)
        conflict = self.conflict(conflict_input)
        sparse_mask = torch.sigmoid(self.sparse_k * (support - conflict - self.sparse_tau))

        delta_input = torch.cat([depth_attn, torch.abs(primary_feat - depth_attn)], dim=1)
        delta = self.residual_adapter(delta_input)
        return primary_feat + self.gamma * sparse_mask * delta


class DFormerGuidedCyclicAttentionFusionPlus(nn.Module):
    def __init__(self, rgb_channels, depth_channels):
        super().__init__()
        hidden = max(rgb_channels // 8, 16)
        self.depth_proj = nn.Sequential(
            nn.Conv2d(depth_channels, rgb_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(rgb_channels),
        )
        self.rectify_gate = nn.Sequential(
            nn.Conv2d(rgb_channels * 3, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=3,
                padding=1,
                groups=hidden,
                bias=False,
            ),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, rgb_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.rectify_update = nn.Sequential(
            nn.Conv2d(rgb_channels * 3, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=3,
                padding=1,
                groups=hidden,
                bias=False,
            ),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, rgb_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(rgb_channels),
        )
        self.query = nn.Sequential(
            nn.Conv2d(rgb_channels, rgb_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(rgb_channels),
        )
        self.key = nn.Sequential(
            nn.Conv2d(rgb_channels, rgb_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(rgb_channels),
        )
        self.value = nn.Sequential(
            nn.Conv2d(rgb_channels, rgb_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(rgb_channels),
        )
        self.relation_logits = nn.Sequential(
            nn.Conv2d(rgb_channels * 4, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, rgb_channels * 2, kernel_size=1),
        )
        self.support = nn.Sequential(
            nn.Conv2d(rgb_channels * 3, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=3,
                padding=1,
                groups=hidden,
                bias=False,
            ),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, rgb_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.conflict = nn.Sequential(
            nn.Conv2d(rgb_channels * 3, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=3,
                padding=1,
                groups=hidden,
                bias=False,
            ),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, rgb_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.support_update = nn.Sequential(
            nn.Conv2d(rgb_channels * 4, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=3,
                padding=1,
                groups=hidden,
                bias=False,
            ),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, rgb_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(rgb_channels),
        )
        self.detail_update = nn.Sequential(
            nn.Conv2d(rgb_channels * 3, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=3,
                padding=1,
                groups=hidden,
                bias=False,
            ),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, rgb_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(rgb_channels),
        )
        self.beta = nn.Parameter(torch.full((1, rgb_channels, 1, 1), 1e-3))
        self.gamma_s = nn.Parameter(torch.full((1, rgb_channels, 1, 1), 1e-3))
        self.gamma_d = nn.Parameter(torch.full((1, rgb_channels, 1, 1), 2e-4))
        self.sparse_k = nn.Parameter(torch.full((1, rgb_channels, 1, 1), 5.0))
        self.sparse_tau = nn.Parameter(torch.zeros(1, rgb_channels, 1, 1))
        self.detail_k = nn.Parameter(torch.full((1, rgb_channels, 1, 1), 3.0))
        self.detail_tau = nn.Parameter(torch.full((1, rgb_channels, 1, 1), 0.1))
        self.noise_token = nn.Parameter(torch.zeros(1, rgb_channels, 1, 1))

    def forward(self, primary_feat, depth_feat):
        depth = self.depth_proj(depth_feat)
        diff0 = torch.abs(primary_feat - depth)
        rectify_input = torch.cat([primary_feat, depth, diff0], dim=1)
        rectify_gate = self.rectify_gate(rectify_input)
        rectify_update = self.rectify_update(rectify_input)
        depth_rect = depth + self.beta * rectify_gate * rectify_update

        query = self.query(primary_feat)
        key = self.key(depth_rect)
        value = self.value(depth_rect)
        relation_input = torch.cat([query, key, torch.abs(query - key), query * key], dim=1)
        relation_logits = self.relation_logits(relation_input)
        batch_size, channels, height, width = primary_feat.shape
        relation_logits = relation_logits.reshape(batch_size, 2, channels, height, width)
        relation = torch.softmax(relation_logits, dim=1)
        r_noise = relation[:, 0]
        r_depth = relation[:, 1]
        depth_rel = r_depth * value + r_noise * self.noise_token

        support_input = torch.cat([primary_feat, depth_rel, primary_feat * depth_rel], dim=1)
        support = self.support(support_input)
        conflict_input = torch.cat([primary_feat, depth_rel, torch.abs(primary_feat - depth_rel)], dim=1)
        conflict = self.conflict(conflict_input)
        support_select = torch.sigmoid(
            self.sparse_k * (support + r_depth - conflict - self.sparse_tau)
        )
        detail_select = torch.sigmoid(
            self.detail_k * (conflict - support - self.detail_tau)
        )

        diff_rel = torch.abs(primary_feat - depth_rel)
        support_update_input = torch.cat(
            [primary_feat, depth_rel, primary_feat * depth_rel, diff_rel],
            dim=1,
        )
        support_update = self.support_update(support_update_input)
        detail_update_input = torch.cat([primary_feat, depth_rel, diff_rel], dim=1)
        detail_update = self.detail_update(detail_update_input)
        return (
            primary_feat
            + self.gamma_s * support_select * support_update
            + self.gamma_d * detail_select * detail_update
        )


class DFormerGuidedCyclicAttentionFusionPlusCSG(nn.Module):
    def __init__(self, rgb_channels, depth_channels):
        super().__init__()
        hidden = max(rgb_channels // 8, 16)
        self.depth_proj = nn.Sequential(
            nn.Conv2d(depth_channels, rgb_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(rgb_channels),
        )
        self.rectify_gate = nn.Sequential(
            nn.Conv2d(rgb_channels * 3, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=3,
                padding=1,
                groups=hidden,
                bias=False,
            ),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, rgb_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.rectify_update = nn.Sequential(
            nn.Conv2d(rgb_channels * 3, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=3,
                padding=1,
                groups=hidden,
                bias=False,
            ),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, rgb_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(rgb_channels),
        )
        self.query = nn.Sequential(
            nn.Conv2d(rgb_channels, rgb_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(rgb_channels),
        )
        self.key = nn.Sequential(
            nn.Conv2d(rgb_channels, rgb_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(rgb_channels),
        )
        self.value = nn.Sequential(
            nn.Conv2d(rgb_channels, rgb_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(rgb_channels),
        )
        self.relation_logits = nn.Sequential(
            nn.Conv2d(rgb_channels * 4, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, rgb_channels * 2, kernel_size=1),
        )
        self.support = nn.Sequential(
            nn.Conv2d(rgb_channels * 3, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=3,
                padding=1,
                groups=hidden,
                bias=False,
            ),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, rgb_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.conflict = nn.Sequential(
            nn.Conv2d(rgb_channels * 3, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=3,
                padding=1,
                groups=hidden,
                bias=False,
            ),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, rgb_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.support_update = nn.Sequential(
            nn.Conv2d(rgb_channels * 4, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=3,
                padding=1,
                groups=hidden,
                bias=False,
            ),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, rgb_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(rgb_channels),
        )
        self.detail_update = nn.Sequential(
            nn.Conv2d(rgb_channels * 3, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=3,
                padding=1,
                groups=hidden,
                bias=False,
            ),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, rgb_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(rgb_channels),
        )
        self.beta = nn.Parameter(torch.full((1, rgb_channels, 1, 1), 1e-3))
        self.gamma_s = nn.Parameter(torch.full((1, rgb_channels, 1, 1), 1e-3))
        self.gamma_d = nn.Parameter(torch.full((1, rgb_channels, 1, 1), 2e-4))
        self.sparse_k = nn.Parameter(torch.full((1, rgb_channels, 1, 1), 5.0))
        self.sparse_tau = nn.Parameter(torch.zeros(1, rgb_channels, 1, 1))
        self.detail_k = nn.Parameter(torch.full((1, rgb_channels, 1, 1), 3.0))
        self.detail_tau = nn.Parameter(torch.full((1, rgb_channels, 1, 1), 0.1))
        self.noise_token = nn.Parameter(torch.zeros(1, rgb_channels, 1, 1))
        self.semantic_alpha = nn.Parameter(torch.full((1, rgb_channels, 1, 1), 1e-3))

    def forward(self, primary_feat, depth_feat, semantic_gate=None):
        depth = self.depth_proj(depth_feat)
        if semantic_gate is not None:
            p_guided = primary_feat * (1 + self.semantic_alpha * (2 * semantic_gate - 1))
        else:
            p_guided = primary_feat

        diff0 = torch.abs(p_guided - depth)
        rectify_input = torch.cat([p_guided, depth, diff0], dim=1)
        rectify_gate = self.rectify_gate(rectify_input)
        rectify_update = self.rectify_update(rectify_input)
        depth_rect = depth + self.beta * rectify_gate * rectify_update

        query = self.query(p_guided)
        key = self.key(depth_rect)
        value = self.value(depth_rect)
        relation_input = torch.cat([query, key, torch.abs(query - key), query * key], dim=1)
        relation_logits = self.relation_logits(relation_input)
        batch_size, channels, height, width = primary_feat.shape
        relation_logits = relation_logits.reshape(batch_size, 2, channels, height, width)
        relation = torch.softmax(relation_logits, dim=1)
        r_noise = relation[:, 0]
        r_depth = relation[:, 1]
        depth_rel = r_depth * value + r_noise * self.noise_token

        support_input = torch.cat([p_guided, depth_rel, p_guided * depth_rel], dim=1)
        support = self.support(support_input)
        conflict_input = torch.cat([p_guided, depth_rel, torch.abs(p_guided - depth_rel)], dim=1)
        conflict = self.conflict(conflict_input)
        support_select = torch.sigmoid(
            self.sparse_k * (support + r_depth - conflict - self.sparse_tau)
        )
        detail_select = torch.sigmoid(
            self.detail_k * (conflict - support - self.detail_tau)
        )

        diff_rel = torch.abs(p_guided - depth_rel)
        support_update_input = torch.cat(
            [p_guided, depth_rel, p_guided * depth_rel, diff_rel],
            dim=1,
        )
        support_update = self.support_update(support_update_input)
        detail_update_input = torch.cat([p_guided, depth_rel, diff_rel], dim=1)
        detail_update = self.detail_update(detail_update_input)
        return (
            primary_feat
            + self.gamma_s * support_select * support_update
            + self.gamma_d * detail_select * detail_update
        )


class DFormerV2PGSparseCompSegmentor(nn.Module):
    def __init__(self, num_classes=40, dformerv2_pretrained=None):
        super().__init__()
        self.rgb_encoder = DFormerv2_S()
        if dformerv2_pretrained:
            self.pretrained_load_stats = load_dformerv2_pretrained(self.rgb_encoder, dformerv2_pretrained)
        else:
            self.pretrained_load_stats = None
        self.depth_encoder = DepthEncoder()
        self.fusions = nn.ModuleList([
            PrimaryGuidedSparseDepthCompensation(rgb_ch, depth_ch)
            for rgb_ch, depth_ch in zip(self.rgb_encoder.out_channels, self.depth_encoder.out_channels)
        ])
        self.decoder = SimpleFPNDecoder(self.rgb_encoder.out_channels, num_classes=num_classes)

    def forward(self, rgb, depth):
        dformer_feats = self.rgb_encoder(rgb, depth)
        depth_feats = self.depth_encoder(depth)

        fused_feats = []
        for fusion, primary_feat, depth_feat in zip(self.fusions, dformer_feats, depth_feats):
            if primary_feat.shape[-2:] != depth_feat.shape[-2:]:
                depth_feat = F.interpolate(
                    depth_feat,
                    size=primary_feat.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            fused_feats.append(fusion(primary_feat, depth_feat))

        return self.decoder(fused_feats, input_size=rgb.shape[-2:])


class LitDFormerV2PGSparseComp(BaseLitSeg):
    def __init__(self, num_classes=40, lr=1e-4, dformerv2_pretrained=None):
        super().__init__(num_classes=num_classes, lr=lr)
        self.model = DFormerV2PGSparseCompSegmentor(
            num_classes=num_classes,
            dformerv2_pretrained=dformerv2_pretrained,
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)


class DFormerV2DGCAFFullSegmentor(nn.Module):
    def __init__(self, num_classes=40, dformerv2_pretrained=None):
        super().__init__()
        self.rgb_encoder = DFormerv2_S()
        if dformerv2_pretrained:
            self.pretrained_load_stats = load_dformerv2_pretrained(self.rgb_encoder, dformerv2_pretrained)
        else:
            self.pretrained_load_stats = None
        self.depth_encoder = DepthEncoder()
        self.fusions = nn.ModuleList([
            DFormerGuidedCyclicAttentionFusion(rgb_ch, depth_ch)
            for rgb_ch, depth_ch in zip(self.rgb_encoder.out_channels, self.depth_encoder.out_channels)
        ])
        self.decoder = SimpleFPNDecoder(self.rgb_encoder.out_channels, num_classes=num_classes)

    def forward(self, rgb, depth):
        dformer_feats = self.rgb_encoder(rgb, depth)
        depth_feats = self.depth_encoder(depth)

        fused_feats = []
        for fusion, primary_feat, depth_feat in zip(self.fusions, dformer_feats, depth_feats):
            if primary_feat.shape[-2:] != depth_feat.shape[-2:]:
                depth_feat = F.interpolate(
                    depth_feat,
                    size=primary_feat.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            fused_feats.append(fusion(primary_feat, depth_feat))

        return self.decoder(fused_feats, input_size=rgb.shape[-2:])


class LitDFormerV2DGCAFFull(BaseLitSeg):
    def __init__(self, num_classes=40, lr=1e-4, dformerv2_pretrained=None):
        super().__init__(num_classes=num_classes, lr=lr)
        self.model = DFormerV2DGCAFFullSegmentor(
            num_classes=num_classes,
            dformerv2_pretrained=dformerv2_pretrained,
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)


class DFormerV2DGCAFPlusSegmentor(nn.Module):
    def __init__(self, num_classes=40, dformerv2_pretrained=None):
        super().__init__()
        self.rgb_encoder = DFormerv2_S()
        if dformerv2_pretrained:
            self.pretrained_load_stats = load_dformerv2_pretrained(self.rgb_encoder, dformerv2_pretrained)
        else:
            self.pretrained_load_stats = None
        self.depth_encoder = DepthEncoder()
        self.fusions = nn.ModuleList([
            DFormerGuidedCyclicAttentionFusionPlus(rgb_ch, depth_ch)
            for rgb_ch, depth_ch in zip(self.rgb_encoder.out_channels, self.depth_encoder.out_channels)
        ])
        self.decoder = SimpleFPNDecoder(self.rgb_encoder.out_channels, num_classes=num_classes)

    def forward(self, rgb, depth):
        dformer_feats = self.rgb_encoder(rgb, depth)
        depth_feats = self.depth_encoder(depth)

        fused_feats = []
        for fusion, primary_feat, depth_feat in zip(self.fusions, dformer_feats, depth_feats):
            if primary_feat.shape[-2:] != depth_feat.shape[-2:]:
                depth_feat = F.interpolate(
                    depth_feat,
                    size=primary_feat.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            fused_feats.append(fusion(primary_feat, depth_feat))

        return self.decoder(fused_feats, input_size=rgb.shape[-2:])


class LitDFormerV2DGCAFPlus(BaseLitSeg):
    def __init__(self, num_classes=40, lr=1e-4, dformerv2_pretrained=None):
        super().__init__(num_classes=num_classes, lr=lr)
        self.model = DFormerV2DGCAFPlusSegmentor(
            num_classes=num_classes,
            dformerv2_pretrained=dformerv2_pretrained,
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)


class DFormerV2DGCAFPlusCSGSegmentor(nn.Module):
    def __init__(self, num_classes=40, dformerv2_pretrained=None):
        super().__init__()
        self.rgb_encoder = DFormerv2_S()
        if dformerv2_pretrained:
            self.pretrained_load_stats = load_dformerv2_pretrained(self.rgb_encoder, dformerv2_pretrained)
        else:
            self.pretrained_load_stats = None
        self.depth_encoder = DepthEncoder()
        self.fusions = nn.ModuleList([
            DFormerGuidedCyclicAttentionFusionPlusCSG(rgb_ch, depth_ch)
            for rgb_ch, depth_ch in zip(self.rgb_encoder.out_channels, self.depth_encoder.out_channels)
        ])
        c4_channels = self.rgb_encoder.out_channels[-1]
        self.semantic_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c4_channels, c4_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c4_channels, c4_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.semantic_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c4_channels, channels, kernel_size=1, bias=False),
                nn.Sigmoid(),
            )
            for channels in self.rgb_encoder.out_channels
        ])
        self.decoder = SimpleFPNDecoder(self.rgb_encoder.out_channels, num_classes=num_classes)

    def forward(self, rgb, depth):
        dformer_feats = self.rgb_encoder(rgb, depth)
        depth_feats = self.depth_encoder(depth)
        global_context = self.semantic_context(dformer_feats[-1])
        semantic_gates = [proj(global_context) for proj in self.semantic_projs]

        fused_feats = []
        for fusion, primary_feat, depth_feat, semantic_gate in zip(
            self.fusions,
            dformer_feats,
            depth_feats,
            semantic_gates,
        ):
            if primary_feat.shape[-2:] != depth_feat.shape[-2:]:
                depth_feat = F.interpolate(
                    depth_feat,
                    size=primary_feat.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            fused_feats.append(fusion(primary_feat, depth_feat, semantic_gate))

        return self.decoder(fused_feats, input_size=rgb.shape[-2:])


class LitDFormerV2DGCAFPlusCSG(BaseLitSeg):
    def __init__(self, num_classes=40, lr=1e-4, dformerv2_pretrained=None):
        super().__init__(num_classes=num_classes, lr=lr)
        self.model = DFormerV2DGCAFPlusCSGSegmentor(
            num_classes=num_classes,
            dformerv2_pretrained=dformerv2_pretrained,
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
