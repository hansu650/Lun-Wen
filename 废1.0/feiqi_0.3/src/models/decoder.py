"""Decoder module."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_relu(in_channels, out_channels, kernel_size=3, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class PPMBlock(nn.Module):
    def __init__(self, in_channels, out_channels=256, branch_channels=64, pool_scales=(1, 2, 3, 6)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(in_channels, branch_channels, 1, bias=False),
                nn.ReLU(inplace=True),
            )
            for scale in pool_scales
        ])
        self.bottleneck = conv_bn_relu(in_channels + len(pool_scales) * branch_channels, out_channels)

    def forward(self, x):
        ppm_feats = [x]
        for branch in self.branches:
            pooled = branch(x)
            ppm_feats.append(F.interpolate(pooled, size=x.shape[-2:], mode="bilinear", align_corners=False))
        return self.bottleneck(torch.cat(ppm_feats, dim=1))


class RCMLiteBlock(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.reduce = conv_bn_relu(channels * 4, channels)
        self.enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=(1, 5), padding=(0, 2), groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=(5, 1), padding=(2, 0), groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        base = self.reduce(x)
        enhance = self.enhance(base)
        return base + self.gamma * enhance


class SimpleFPNDecoder(nn.Module):
    def __init__(self, in_channels, out_channels=256, num_classes=40):
        super().__init__()
        self.lateral1 = conv_bn_relu(in_channels[0], out_channels, kernel_size=1, padding=0)
        self.lateral2 = conv_bn_relu(in_channels[1], out_channels, kernel_size=1, padding=0)
        self.lateral3 = conv_bn_relu(in_channels[2], out_channels, kernel_size=1, padding=0)
        self.ppm4 = PPMBlock(in_channels[3], out_channels=out_channels)

        self.smooth1 = conv_bn_relu(out_channels, out_channels)
        self.smooth2 = conv_bn_relu(out_channels, out_channels)
        self.smooth3 = conv_bn_relu(out_channels, out_channels)
        self.smooth4 = conv_bn_relu(out_channels, out_channels)

        self.reconstruct = RCMLiteBlock(out_channels)
        self.classifier = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(out_channels, num_classes, 1),
        )

    def forward(self, features, input_size):
        c1, c2, c3, c4 = features

        lat1 = self.lateral1(c1)
        lat2 = self.lateral2(c2)
        lat3 = self.lateral3(c3)
        p4 = self.ppm4(c4)

        p3 = lat3 + F.interpolate(p4, size=lat3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = lat2 + F.interpolate(p3, size=lat2.shape[-2:], mode="bilinear", align_corners=False)
        p1 = lat1 + F.interpolate(p2, size=lat1.shape[-2:], mode="bilinear", align_corners=False)

        p1 = self.smooth1(p1)
        p2 = self.smooth2(p2)
        p3 = self.smooth3(p3)
        p4 = self.smooth4(p4)

        up_p2 = F.interpolate(p2, size=p1.shape[-2:], mode="bilinear", align_corners=False)
        up_p3 = F.interpolate(p3, size=p1.shape[-2:], mode="bilinear", align_corners=False)
        up_p4 = F.interpolate(p4, size=p1.shape[-2:], mode="bilinear", align_corners=False)
        fuse = torch.cat([p1, up_p2, up_p3, up_p4], dim=1)

        out = self.reconstruct(fuse)
        logits = self.classifier(out)
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
