"""Archived mid-fusion experiment variants removed from active mainline.

These classes are intentionally lightweight archives of the ideas, not active
training modules. Use the experiment reports and branch history for exact
runnable snapshots.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def depth01_from_official_depth(depth):
    """R019/R020 adapter used to reconstruct approximate [0, 1] depth."""
    return torch.clamp(depth * 0.28 + 0.48, min=0.0, max=1.0)


class BranchDepthAdapterArchive:
    summary = {
        "runs": ["R019", "R020"],
        "best_val_miou": {"R019": 0.532539, "R020": 0.532924},
        "decision": "archive; below R016 and unstable",
    }

    @staticmethod
    def blend_depth(depth, alpha=0.05):
        return (1.0 - alpha) * depth + alpha * depth01_from_official_depth(depth)


class DepthEncoderBNEvalArchive:
    summary = {
        "run": "R025",
        "best_val_miou": 0.532572,
        "decision": "archive; late collapse",
    }

    @staticmethod
    def freeze_depth_encoder_bn_train_mode(depth_encoder):
        for module in depth_encoder.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


class OfficialInitLocalModulesArchive:
    summary = {
        "run": "R026",
        "best_val_miou": 0.507906,
        "decision": "archive; strongly negative",
    }

    @staticmethod
    def init_official_style_local_modules(module):
        for child in module.modules():
            if isinstance(child, nn.Conv2d):
                nn.init.kaiming_normal_(child.weight, mode="fan_in", nonlinearity="relu")
                if child.bias is not None:
                    nn.init.zeros_(child.bias)
            elif isinstance(child, nn.BatchNorm2d):
                child.eps = 1e-3
                child.momentum = 0.1
                nn.init.ones_(child.weight)
                nn.init.zeros_(child.bias)


class PrimaryResidualDepthInjection(nn.Module):
    summary = {
        "run": "R027",
        "best_val_miou": 0.536739,
        "decision": "archive; high peak but unstable and below R016",
    }

    def __init__(self, rgb_channels, depth_channels):
        super().__init__()
        self.depth_proj = nn.Conv2d(depth_channels, rgb_channels, 1)
        self.residual = nn.Sequential(
            nn.Conv2d(rgb_channels * 2, rgb_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(rgb_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(rgb_channels, rgb_channels, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, rgb_feat, depth_feat):
        d = self.depth_proj(depth_feat)
        residual = self.residual(torch.cat([d, torch.abs(rgb_feat - d)], dim=1))
        return rgb_feat + residual


class GatedFusionResidualTop(nn.Module):
    summary = {
        "run": "R030",
        "best_val_miou": 0.536454,
        "decision": "archive; below R016/R027",
    }

    def __init__(self, base_fusion, rgb_channels):
        super().__init__()
        self.base_fusion = base_fusion
        self.residual = nn.Sequential(
            nn.Conv2d(rgb_channels * 4, rgb_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(rgb_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(rgb_channels, rgb_channels, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, rgb_feat, depth_feat):
        d = self.base_fusion.depth_proj(depth_feat)
        g = self.base_fusion.gate(torch.cat([rgb_feat, d], dim=1))
        base = self.base_fusion.refine(g * rgb_feat + (1 - g) * d)
        residual = self.residual(torch.cat([rgb_feat, d, base, torch.abs(rgb_feat - d)], dim=1))
        return base + residual
