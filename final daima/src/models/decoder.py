"""Active decoder modules."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleFPNDecoder(nn.Module):
    """Minimal FPN decoder used by the active baseline and TGGA/PMAD paths."""

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


class FrequencyAwareTopDownFuse(nn.Module):
    def __init__(self, channels, hidden_channels=64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 4, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels * 2, 1),
        )
        self.low_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.high_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.gamma_low = nn.Parameter(torch.tensor(0.05))
        self.gamma_high = nn.Parameter(torch.tensor(0.05))
        nn.init.zeros_(self.low_proj.weight)
        nn.init.zeros_(self.high_proj.weight)

    @staticmethod
    def lowpass(x, kernel_size):
        return F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, hr_feat, lr_feat):
        lr_up = F.interpolate(lr_feat, size=hr_feat.shape[-2:], mode="bilinear", align_corners=False)
        hr_low = self.lowpass(hr_feat, kernel_size=5)
        lr_low = self.lowpass(lr_up, kernel_size=5)
        hr_high = hr_feat - self.lowpass(hr_feat, kernel_size=3)
        lr_high = lr_up - self.lowpass(lr_up, kernel_size=3)

        gate_input = torch.cat([hr_feat, lr_up, torch.abs(hr_feat - lr_up), hr_high - lr_high], dim=1)
        gate_low, gate_high = self.gate(gate_input).chunk(2, dim=1)
        gate_low = torch.sigmoid(gate_low)
        gate_high = torch.sigmoid(gate_high)

        low_corr = self.low_proj(lr_low - hr_low)
        high_corr = self.high_proj(hr_high - lr_high)
        return hr_feat + lr_up + self.gamma_low * gate_low * low_corr + self.gamma_high * gate_high * high_corr


class FrequencyAwareFPNDecoder(nn.Module):
    def __init__(self, in_channels, out_channels=128, num_classes=40):
        super().__init__()
        self.lateral4 = nn.Conv2d(in_channels[3], out_channels, 1)
        self.lateral3 = nn.Conv2d(in_channels[2], out_channels, 1)
        self.lateral2 = nn.Conv2d(in_channels[1], out_channels, 1)
        self.lateral1 = nn.Conv2d(in_channels[0], out_channels, 1)

        self.fuse43 = FrequencyAwareTopDownFuse(out_channels)
        self.fuse32 = FrequencyAwareTopDownFuse(out_channels)
        self.fuse21 = FrequencyAwareTopDownFuse(out_channels)

        self.smooth = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(out_channels, num_classes, 1)

    def forward(self, features, input_size):
        c1, c2, c3, c4 = features

        p4 = self.lateral4(c4)
        p3 = self.fuse43(self.lateral3(c3), p4)
        p2 = self.fuse32(self.lateral2(c2), p3)
        p1 = self.fuse21(self.lateral1(c1), p2)

        p1 = self.smooth(p1)
        p1 = F.interpolate(p1, size=input_size, mode="bilinear", align_corners=False)
        return self.classifier(p1)
