"""FFT-based modality-wise frequency enhancement blocks."""
import torch
import torch.nn as nn


class FrequencyClean(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        nn.init.normal_(self.conv.weight, std=1e-4)

    def forward(self, high):
        return self.conv(high)


class FFTFrequencyEnhance(nn.Module):
    def __init__(self, channels, cutoff_ratio=0.25, gamma_init=0.05):
        super().__init__()
        if not 0 < cutoff_ratio <= 0.5:
            raise ValueError("cutoff_ratio must be in (0, 0.5]")
        self.cutoff_ratio = float(cutoff_ratio)
        self.gate_conv = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.clean = FrequencyClean(channels)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def _circular_lowpass_mask(self, H, W, device, dtype):
        cy = (H - 1) / 2.0
        cx = (W - 1) / 2.0
        y = torch.arange(H, device=device, dtype=dtype) - cy
        x = torch.arange(W, device=device, dtype=dtype) - cx
        Y, X = torch.meshgrid(y, x, indexing="ij")
        dist = torch.sqrt(X ** 2 + Y ** 2)
        radius = self.cutoff_ratio * min(H, W) / 2.0
        return (dist <= radius).view(1, 1, H, W)

    def forward(self, x):
        _, _, H, W = x.shape
        mask = self._circular_lowpass_mask(H, W, x.device, x.dtype)

        X = torch.fft.fft2(x, dim=(-2, -1))
        X_shift = torch.fft.fftshift(X, dim=(-2, -1))
        X_low = X_shift * mask
        x_low = torch.fft.ifft2(
            torch.fft.ifftshift(X_low, dim=(-2, -1)),
            dim=(-2, -1),
        ).real

        x_high = x - x_low
        gate = torch.sigmoid(self.gate_conv(torch.cat([x, x_high], dim=1)))
        clean_high = self.clean(x_high)
        return x + self.gamma * gate * clean_high
