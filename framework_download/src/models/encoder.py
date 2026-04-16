"""Encoder modules for RGB, depth, and early-fusion variants."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model, SwinModel
from torchvision.models import ResNet18_Weights, resnet18


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PRETRAINED_ROOT = PROJECT_ROOT / "pretrained"
SWIN_BASE_DIR = PRETRAINED_ROOT / "swin_base"
DINOV2_BASE_DIR = PRETRAINED_ROOT / "dinov2_base"


def _require_local_model_dir(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Local pretrained directory not found: {path}. "
            "Please place the model under the expected pretrained/ folder."
        )
    return path


def _compute_pyramid_sizes(height: int, width: int) -> list[tuple[int, int]]:
    # Swin-B produces stage outputs at roughly 1/4, 1/8, 1/16, 1/32 of input.
    # We mirror those sizes for the depth branch so the existing gated fusion
    # and FPN decoder can keep their original assumptions.
    return [
        (max(1, height // 4), max(1, width // 4)),
        (max(1, height // 8), max(1, width // 8)),
        (max(1, height // 16), max(1, width // 16)),
        (max(1, height // 32), max(1, width // 32)),
    ]


class DinoStageNeck(nn.Module):
    """Project one DINO hidden state into a CNN-style stage feature map.

    Each stage does two jobs:
    1. 1x1 conv maps DINO's shared 768-dim token space into the target stage width
    2. bilinear resize aligns the spatial size with the Swin pyramid stage
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        x = self.proj(x)
        if x.shape[-2:] != target_size:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return x


class SwinBackboneEncoder(nn.Module):
    """RGB branch backed by the local Swin-B checkpoint."""

    def __init__(self, model_dir: Path | None = None) -> None:
        super().__init__()
        model_dir = _require_local_model_dir(model_dir or SWIN_BASE_DIR)
        self.backbone = SwinModel.from_pretrained(str(model_dir))

        # The first four reshaped_hidden_states already form a natural FPN pyramid:
        # stage1: (B, 128, H/4,  W/4)
        # stage2: (B, 256, H/8,  W/8)
        # stage3: (B, 512, H/16, W/16)
        # stage4: (B, 1024,H/32, W/32)
        self.out_channels = [128, 256, 512, 1024]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = self.backbone(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True,
        )
        return list(outputs.reshaped_hidden_states[:4])


class Dinov2DepthBackbone(nn.Module):
    """Depth branch backed by the local DINOv2-B checkpoint.

    DINOv2 returns token sequences, not a hierarchical CNN pyramid, so we:
    1. copy 1-channel depth into 3 channels
    2. take a few intermediate hidden states
    3. remove the CLS token and reshape tokens back to (B, C, H, W)
    4. use a lightweight neck to project and resize them into 4 stage features
    """

    def __init__(
        self,
        model_dir: Path | None = None,
        hidden_state_indices: tuple[int, int, int, int] = (3, 6, 9, 12),
    ) -> None:
        super().__init__()
        model_dir = _require_local_model_dir(model_dir or DINOV2_BASE_DIR)
        self.backbone = Dinov2Model.from_pretrained(str(model_dir))
        self.hidden_state_indices = hidden_state_indices
        self.patch_size = int(self.backbone.config.patch_size)

        # We align the final stage widths with Swin so the old gated fusion can
        # still zip RGB/depth stages together without any train.py changes.
        self.out_channels = [128, 256, 512, 1024]
        self.stage_necks = nn.ModuleList(
            [
                DinoStageNeck(self.backbone.config.hidden_size, out_channels)
                for out_channels in self.out_channels
            ]
        )

    def _repeat_depth_to_rgb(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.shape[1] == 1:
            return x.repeat(1, 3, 1, 1)
        if x.shape[1] == 3:
            return x
        raise ValueError(f"Depth input must have 1 or 3 channels, got {x.shape[1]}.")

    def _tokens_to_map(
        self,
        tokens: torch.Tensor,
        input_height: int,
        input_width: int,
    ) -> torch.Tensor:
        batch_size, token_count, channels = tokens.shape
        grid_h = max(1, input_height // self.patch_size)
        grid_w = max(1, input_width // self.patch_size)

        # DINO keeps one CLS token in front; for dense prediction we discard it.
        if token_count == grid_h * grid_w + 1:
            tokens = tokens[:, 1:, :]
            token_count = tokens.shape[1]

        if token_count != grid_h * grid_w:
            raise ValueError(
                f"DINO token count {token_count} does not match grid {grid_h}x{grid_w}."
            )

        return tokens.transpose(1, 2).reshape(batch_size, channels, grid_h, grid_w)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self._repeat_depth_to_rgb(x)
        input_height, input_width = x.shape[-2:]

        outputs = self.backbone(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True,
        )

        target_sizes = _compute_pyramid_sizes(input_height, input_width)
        features: list[torch.Tensor] = []

        for neck, hidden_idx, target_size in zip(
            self.stage_necks, self.hidden_state_indices, target_sizes
        ):
            hidden = outputs.hidden_states[hidden_idx]
            feature_map = self._tokens_to_map(hidden, input_height, input_width)
            features.append(neck(feature_map, target_size))

        return features


class RGBEncoder(SwinBackboneEncoder):
    """RGB encoder using the local Swin-B checkpoint."""

    def __init__(self) -> None:
        super().__init__(model_dir=SWIN_BASE_DIR)


class DepthEncoder(Dinov2DepthBackbone):
    """Depth encoder using the local DINOv2-B checkpoint."""

    def __init__(self) -> None:
        super().__init__(model_dir=DINOV2_BASE_DIR)


class EarlyFusionEncoder(nn.Module):
    """Fallback early-fusion encoder that keeps the original 4-channel ResNet."""

    def __init__(self) -> None:
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

        old_conv = resnet.conv1
        new_conv = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv.weight[:, :3].copy_(old_conv.weight)
            new_conv.weight[:, 3].copy_(old_conv.weight.mean(dim=1))
        resnet.conv1 = new_conv

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.out_channels = [64, 128, 256, 512]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = []
        x = self.layer0(x)
        x = self.layer1(x)
        feats.append(x)
        x = self.layer2(x)
        feats.append(x)
        x = self.layer3(x)
        feats.append(x)
        x = self.layer4(x)
        feats.append(x)
        return feats
