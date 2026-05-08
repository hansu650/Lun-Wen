"""Training-only multi-scale frequency covariance loss."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleFrequencyCovarianceLoss(nn.Module):
    def __init__(
        self,
        rgb_channels,
        depth_channels,
        proj_dim=64,
        kernel_size=3,
        eta=1.0,
        stage_weights=(1.0, 1.0, 1.0, 1.0),
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("freq_kernel_size must be odd")
        if len(rgb_channels) != 4 or len(depth_channels) != 4 or len(stage_weights) != 4:
            raise ValueError("rgb_channels, depth_channels, and stage_weights must have length 4")

        self.kernel_size = kernel_size
        self.eta = float(eta)
        self.rgb_projectors = nn.ModuleList([
            nn.Conv2d(ch, proj_dim, kernel_size=1, bias=False)
            for ch in rgb_channels
        ])
        self.depth_projectors = nn.ModuleList([
            nn.Conv2d(ch, proj_dim, kernel_size=1, bias=False)
            for ch in depth_channels
        ])
        self.register_buffer(
            "stage_weights",
            torch.tensor(stage_weights, dtype=torch.float32),
            persistent=False,
        )

    def _split_frequency(self, x):
        low = F.avg_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
        )
        high = x - low
        return high, low

    def _covariance(self, z):
        c = z.shape[1]
        z = z.permute(0, 2, 3, 1).reshape(-1, c)
        n = z.shape[0]
        if n <= 1:
            return z.new_zeros((c, c))
        z = z - z.mean(dim=0, keepdim=True)
        return z.t().matmul(z) / (n - 1)

    def _covariance_loss(self, rgb_feat, depth_feat, rgb_projector, depth_projector):
        rgb_z = rgb_projector(rgb_feat)
        depth_z = depth_projector(depth_feat)
        rgb_cov = self._covariance(rgb_z)
        depth_cov = self._covariance(depth_z)
        return F.mse_loss(rgb_cov, depth_cov)

    def forward(self, rgb_feats, depth_feats):
        total = rgb_feats[0].new_zeros(())
        high_total = rgb_feats[0].new_zeros(())
        low_total = rgb_feats[0].new_zeros(())
        loss_dict = {}
        weights = self.stage_weights.to(device=rgb_feats[0].device, dtype=rgb_feats[0].dtype)

        for idx, (rgb_feat, depth_feat, rgb_projector, depth_projector, weight) in enumerate(
            zip(rgb_feats, depth_feats, self.rgb_projectors, self.depth_projectors, weights),
            start=1,
        ):
            rgb_high, rgb_low = self._split_frequency(rgb_feat)
            depth_high, depth_low = self._split_frequency(depth_feat)

            high_loss = self._covariance_loss(rgb_high, depth_high, rgb_projector, depth_projector)
            low_loss = self._covariance_loss(rgb_low, depth_low, rgb_projector, depth_projector)
            stage_loss = high_loss + self.eta * low_loss

            total = total + weight * stage_loss
            high_total = high_total + weight * high_loss
            low_total = low_total + weight * low_loss
            loss_dict[f"freqcov/stage{idx}"] = stage_loss.detach()

        weight_sum = weights.sum().clamp_min(1.0)
        total = total / weight_sum
        high_total = high_total / weight_sum
        low_total = low_total / weight_sum

        loss_dict["freqcov/total"] = total.detach()
        loss_dict["freqcov/high"] = high_total.detach()
        loss_dict["freqcov/low"] = low_total.detach()
        return total, loss_dict
