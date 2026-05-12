"""Archived decoder experiments removed from the default training path."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PPMContextBlock(nn.Module):
    def __init__(self, in_channels: int, pool_scales=(1, 2, 3, 6), branch_channels=None, alpha_init: float = 0.1):
        super().__init__()
        if branch_channels is None:
            branch_channels = max(in_channels // 4, 64)
        self.pool_scales = tuple(pool_scales)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
            )
            for scale in self.pool_scales
        ])
        context_channels = in_channels + len(self.pool_scales) * branch_channels
        self.fuse = nn.Sequential(
            nn.Conv2d(context_channels, in_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, x):
        pooled = [
            F.interpolate(branch(x), size=x.shape[-2:], mode="bilinear", align_corners=False)
            for branch in self.branches
        ]
        context = self.fuse(torch.cat([x] + pooled, dim=1))
        return x + self.alpha * context


class ContextFPNDecoder(nn.Module):
    def __init__(self, in_channels, out_channels=128, num_classes=40):
        super().__init__()
        self.context = PPMContextBlock(in_channels[3])
        self.lateral4 = nn.Conv2d(in_channels[3], out_channels, 1)
        self.lateral3 = nn.Conv2d(in_channels[2], out_channels, 1)
        self.lateral2 = nn.Conv2d(in_channels[1], out_channels, 1)
        self.lateral1 = nn.Conv2d(in_channels[0], out_channels, 1)

        self.smooth = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(out_channels, num_classes, 1)

    def forward(self, features, input_size):
        c1, c2, c3, c4 = features

        c4 = self.context(c4)
        p4 = self.lateral4(c4)
        p3 = self.lateral3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.lateral2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        p1 = self.lateral1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode="bilinear", align_corners=False)

        p1 = self.smooth(p1)
        p1 = F.interpolate(p1, size=input_size, mode="bilinear", align_corners=False)
        return self.classifier(p1)


class ClassContextBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_classes: int,
        context_channels: int = 64,
        alpha_init: float = 0.1,
        alpha_max: float = 0.2,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps
        self.query_proj = nn.Conv2d(channels, context_channels, kernel_size=1, bias=True)
        self.key_proj = nn.Linear(channels, context_channels)
        self.value_proj = nn.Linear(channels, channels)
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.alpha_max = float(alpha_max)
        raw_alpha = math.log(float(alpha_init) / (self.alpha_max - float(alpha_init)))
        self.raw_alpha = nn.Parameter(torch.tensor(float(raw_alpha)))
        self.scale = context_channels ** -0.5

    @property
    def alpha(self):
        return self.alpha_max * torch.sigmoid(self.raw_alpha)

    def forward(self, p1, aux_logits):
        b, c, h, w = p1.shape
        prob = F.softmax(aux_logits, dim=1)
        prob_flat = prob.flatten(2)
        feat_flat = p1.flatten(2).transpose(1, 2)

        denom = prob_flat.sum(dim=2, keepdim=True).clamp_min(self.eps)
        class_context = torch.bmm(prob_flat, feat_flat) / denom

        query = self.query_proj(p1).flatten(2).transpose(1, 2)
        key = self.key_proj(class_context)
        value = self.value_proj(class_context)

        attn = torch.bmm(query, key.transpose(1, 2)) * self.scale
        attn = F.softmax(attn, dim=2)
        context = torch.bmm(attn, value).transpose(1, 2).reshape(b, c, h, w)

        refined = self.fuse_conv(torch.cat([p1, context], dim=1))
        return p1 + self.alpha * refined


class ClassContextFPNDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=128,
        num_classes=40,
        context_channels=64,
        alpha_init: float = 0.1,
        alpha_max: float = 0.2,
    ):
        super().__init__()
        self.lateral4 = nn.Conv2d(in_channels[3], out_channels, 1)
        self.lateral3 = nn.Conv2d(in_channels[2], out_channels, 1)
        self.lateral2 = nn.Conv2d(in_channels[1], out_channels, 1)
        self.lateral1 = nn.Conv2d(in_channels[0], out_channels, 1)

        self.smooth = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.aux_classifier = nn.Conv2d(out_channels, num_classes, 1)
        self.class_context_block = ClassContextBlock(
            channels=out_channels,
            num_classes=num_classes,
            context_channels=context_channels,
            alpha_init=alpha_init,
            alpha_max=alpha_max,
        )
        self.classifier = nn.Conv2d(out_channels, num_classes, 1)

    def forward(self, features, input_size, return_aux=False):
        c1, c2, c3, c4 = features

        p4 = self.lateral4(c4)
        p3 = self.lateral3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.lateral2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        p1 = self.lateral1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode="bilinear", align_corners=False)

        p1 = self.smooth(p1)
        aux_logits_low = self.aux_classifier(p1)
        p1_refined = self.class_context_block(p1, aux_logits_low)
        final_logits_low = self.classifier(p1_refined)

        final_logits = F.interpolate(final_logits_low, size=input_size, mode="bilinear", align_corners=False)
        if not return_aux:
            return final_logits
        aux_logits = F.interpolate(aux_logits_low, size=input_size, mode="bilinear", align_corners=False)
        return final_logits, aux_logits


class SGBRBlock(nn.Module):
    def __init__(self, channels: int, beta_init: float = 0.05, beta_max: float = 0.2, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.beta_max = float(beta_max)
        raw_beta = math.log(float(beta_init) / (self.beta_max - float(beta_init)))
        self.raw_beta = nn.Parameter(torch.tensor(float(raw_beta)))
        self.residual = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

    @property
    def beta(self):
        return self.beta_max * torch.sigmoid(self.raw_beta)

    def _depth_edge(self, depth, target_size):
        if depth.shape[-2:] != target_size:
            depth = F.interpolate(depth, size=target_size, mode="bilinear", align_corners=False)
        gx = F.conv2d(depth, self.sobel_x.to(dtype=depth.dtype), padding=1)
        gy = F.conv2d(depth, self.sobel_y.to(dtype=depth.dtype), padding=1)
        edge = torch.sqrt(gx.square() + gy.square() + self.eps)
        b = edge.shape[0]
        edge_flat = edge.flatten(1)
        edge_min = edge_flat.min(dim=1).values.view(b, 1, 1, 1)
        edge_max = edge_flat.max(dim=1).values.view(b, 1, 1, 1)
        return (edge - edge_min) / (edge_max - edge_min).clamp_min(self.eps)

    def forward(self, p1, depth, aux_logits):
        prob = F.softmax(aux_logits, dim=1)
        confidence = prob.max(dim=1, keepdim=True).values
        uncertainty = 1.0 - confidence
        depth_edge = self._depth_edge(depth, p1.shape[-2:])
        gate = (uncertainty * depth_edge).clamp(0.0, 1.0)
        residual = self.residual(p1)
        return p1 + self.beta * gate * residual


class SGBRFPNDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=128,
        num_classes=40,
        beta_init: float = 0.05,
        beta_max: float = 0.2,
    ):
        super().__init__()
        self.lateral4 = nn.Conv2d(in_channels[3], out_channels, 1)
        self.lateral3 = nn.Conv2d(in_channels[2], out_channels, 1)
        self.lateral2 = nn.Conv2d(in_channels[1], out_channels, 1)
        self.lateral1 = nn.Conv2d(in_channels[0], out_channels, 1)

        self.smooth = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.aux_classifier = nn.Conv2d(out_channels, num_classes, 1)
        self.sgbr_block = SGBRBlock(channels=out_channels, beta_init=beta_init, beta_max=beta_max)
        self.classifier = nn.Conv2d(out_channels, num_classes, 1)

    def forward(self, features, depth, input_size, return_aux=False):
        c1, c2, c3, c4 = features

        p4 = self.lateral4(c4)
        p3 = self.lateral3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.lateral2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        p1 = self.lateral1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode="bilinear", align_corners=False)

        p1 = self.smooth(p1)
        aux_logits_low = self.aux_classifier(p1)
        p1_refined = self.sgbr_block(p1, depth, aux_logits_low)
        final_logits_low = self.classifier(p1_refined)

        final_logits = F.interpolate(final_logits_low, size=input_size, mode="bilinear", align_corners=False)
        if not return_aux:
            return final_logits
        aux_logits = F.interpolate(aux_logits_low, size=input_size, mode="bilinear", align_corners=False)
        return final_logits, aux_logits
