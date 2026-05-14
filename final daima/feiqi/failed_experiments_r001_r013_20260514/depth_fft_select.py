"""Depth encoder with internal FFT low/high frequency selection."""
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class DepthFrequencySelect(nn.Module):
    def __init__(self, channels, cutoff_ratio=0.30):
        super().__init__()
        self.cutoff_ratio = float(cutoff_ratio)
        self.low_select = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=True,
        )
        self.high_select = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=True,
        )
        nn.init.zeros_(self.low_select.weight)
        nn.init.zeros_(self.low_select.bias)
        nn.init.zeros_(self.high_select.weight)
        nn.init.zeros_(self.high_select.bias)

    def _circular_lowpass_mask(self, H, W, device, dtype):
        cy = (H - 1) / 2.0
        cx = (W - 1) / 2.0
        y = torch.arange(H, device=device, dtype=dtype) - cy
        x = torch.arange(W, device=device, dtype=dtype) - cx
        Y, X = torch.meshgrid(y, x, indexing="ij")
        dist = torch.sqrt(X ** 2 + Y ** 2)
        radius = self.cutoff_ratio * min(H, W) / 2.0
        return (dist <= radius).view(1, 1, H, W)

    def forward(self, depth_feat):
        _, _, H, W = depth_feat.shape
        mask = self._circular_lowpass_mask(H, W, depth_feat.device, depth_feat.dtype)

        X = torch.fft.fft2(depth_feat, dim=(-2, -1))
        X_shift = torch.fft.fftshift(X, dim=(-2, -1))
        X_low = X_shift * mask
        depth_low = torch.fft.ifft2(
            torch.fft.ifftshift(X_low, dim=(-2, -1)),
            dim=(-2, -1),
        ).real
        depth_high = depth_feat - depth_low

        low_weight = torch.sigmoid(self.low_select(depth_feat)) * 2.0
        high_weight = torch.sigmoid(self.high_select(depth_feat)) * 2.0
        return depth_feat + (low_weight - 1.0) * depth_low + (high_weight - 1.0) * depth_high


class DepthEncoderFFTSelect(nn.Module):
    def __init__(self, cutoff_ratio=0.30):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

        old_conv = resnet.conv1
        new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        resnet.conv1 = new_conv

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.fft_select_c2 = DepthFrequencySelect(128, cutoff_ratio=cutoff_ratio)
        self.fft_select_c3 = DepthFrequencySelect(256, cutoff_ratio=cutoff_ratio)
        self.fft_select_c4 = DepthFrequencySelect(512, cutoff_ratio=cutoff_ratio)
        self.out_channels = [64, 128, 256, 512]

    def forward(self, x):
        feats = []
        x = self.layer0(x)
        x = self.layer1(x)
        feats.append(x)
        x = self.layer2(x)
        x = self.fft_select_c2(x)
        feats.append(x)
        x = self.layer3(x)
        x = self.fft_select_c3(x)
        feats.append(x)
        x = self.layer4(x)
        x = self.fft_select_c4(x)
        feats.append(x)
        return feats
