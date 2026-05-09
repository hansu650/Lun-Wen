"""FFT low/high dual-band residual enhancement blocks."""
import math

import torch
import torch.nn as nn


def _inverse_sigmoid(p):
    return math.log(p / (1.0 - p))


class FFTHiLoEnhance(nn.Module):
    def __init__(
        self,
        channels,
        cutoff_ratio=0.25,
        alpha_low_init=0.03,
        alpha_high_init=0.10,
        alpha_max=0.5,
    ):
        super().__init__()
        if not 0.0 < alpha_low_init < alpha_max:
            raise ValueError("alpha_low_init must be in (0, alpha_max)")
        if not 0.0 < alpha_high_init < alpha_max:
            raise ValueError("alpha_high_init must be in (0, alpha_max)")
        self.cutoff_ratio = float(cutoff_ratio)
        self.alpha_max = float(alpha_max)

        raw_alpha_low = _inverse_sigmoid(alpha_low_init / alpha_max)
        raw_alpha_high = _inverse_sigmoid(alpha_high_init / alpha_max)
        self.raw_alpha_low = nn.Parameter(torch.tensor(float(raw_alpha_low)))
        self.raw_alpha_high = nn.Parameter(torch.tensor(float(raw_alpha_high)))

        self.gate_low = nn.Conv2d(channels * 3, channels, kernel_size=1, bias=True)
        self.gate_high = nn.Conv2d(channels * 3, channels, kernel_size=1, bias=True)
        self.clean_low = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.clean_high = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        nn.init.zeros_(self.gate_low.weight)
        nn.init.zeros_(self.gate_low.bias)
        nn.init.zeros_(self.gate_high.weight)
        nn.init.zeros_(self.gate_high.bias)
        nn.init.normal_(self.clean_low.weight, mean=0.0, std=1e-4)
        nn.init.normal_(self.clean_high.weight, mean=0.0, std=1e-4)

    def current_alpha_low(self):
        return self.alpha_max * torch.sigmoid(self.raw_alpha_low)

    def current_alpha_high(self):
        return self.alpha_max * torch.sigmoid(self.raw_alpha_high)

    def _circular_lowpass_mask(self, H, W, device, dtype):
        cy = (H - 1) / 2.0
        cx = (W - 1) / 2.0
        y = torch.arange(H, device=device, dtype=dtype) - cy
        x = torch.arange(W, device=device, dtype=dtype) - cx
        Y, X = torch.meshgrid(y, x, indexing="ij")
        dist = torch.sqrt(X ** 2 + Y ** 2)
        radius = self.cutoff_ratio * min(H, W) / 2.0
        return (dist <= radius).view(1, 1, H, W)

    def _decompose(self, x):
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
        return x_low, x_high

    def forward(self, x):
        x_low, x_high = self._decompose(x)
        gate_input = torch.cat([x, x_low, x_high], dim=1)
        gate_low = torch.sigmoid(self.gate_low(gate_input))
        gate_high = torch.sigmoid(self.gate_high(gate_input))

        low_res = self.current_alpha_low() * gate_low * self.clean_low(x_low)
        high_res = self.current_alpha_high() * gate_high * self.clean_high(x_high)
        return x + low_res + high_res
