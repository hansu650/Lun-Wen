"""解码器模块"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleFPNDecoder(nn.Module):
    """简化版 FPN 解码器"""
    def __init__(self, in_channels, out_channels=128, num_classes=40):
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
        self.classifier = nn.Conv2d(out_channels, num_classes, 1)
    
    def forward(self, features, input_size):
        c1, c2, c3, c4 = features
        
        p4 = self.lateral4(c4)
        p3 = self.lateral3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.lateral2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        p1 = self.lateral1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode="bilinear", align_corners=False)
        
        p1 = self.smooth(p1)
        p1 = F.interpolate(p1, size=input_size, mode="bilinear", align_corners=False)
        return self.classifier(p1)


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
