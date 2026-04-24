from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model, SwinModel


ROOT = Path(__file__).resolve().parents[3]
DINOV2_SMALL_DIR = ROOT / "pretrained" / "dinov2_small"
SWIN_TINY_DIR = ROOT / "pretrained" / "swin_tiny"
STAGE_CHANNELS = [96, 192, 384, 768]
HIDDEN_STATE_INDICES = (3, 6, 9, 12)


def _require_dir(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing pretrained directory: {path}")
    return path


def _target_sizes(height: int, width: int) -> list[tuple[int, int]]:
    return [
        (max(1, height // 4), max(1, width // 4)),
        (max(1, height // 8), max(1, width // 8)),
        (max(1, height // 16), max(1, width // 16)),
        (max(1, height // 32), max(1, width // 32)),
    ]


class DinoStageProjector(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        x = self.proj(x)
        if x.shape[-2:] != size:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return x


class RGBEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Dinov2Model.from_pretrained(str(_require_dir(DINOV2_SMALL_DIR)))
        self.patch_size = int(self.backbone.config.patch_size)
        self.out_channels = STAGE_CHANNELS
        self.projectors = nn.ModuleList(
            [DinoStageProjector(self.backbone.config.hidden_size, out_channels) for out_channels in self.out_channels]
        )

    def _tokens_to_map(self, tokens: torch.Tensor, input_height: int, input_width: int) -> torch.Tensor:
        batch_size, token_count, channels = tokens.shape
        grid_h = max(1, input_height // self.patch_size)
        grid_w = max(1, input_width // self.patch_size)

        if token_count == grid_h * grid_w + 1:
            tokens = tokens[:, 1:, :]
            token_count = tokens.shape[1]

        if token_count != grid_h * grid_w:
            raise ValueError(f"DINO token count {token_count} does not match grid {grid_h}x{grid_w}")

        return tokens.transpose(1, 2).reshape(batch_size, channels, grid_h, grid_w)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        input_height, input_width = x.shape[-2:]
        outputs = self.backbone(pixel_values=x, output_hidden_states=True, return_dict=True)

        features = []
        for projector, hidden_idx, size in zip(self.projectors, HIDDEN_STATE_INDICES, _target_sizes(input_height, input_width)):
            feat = self._tokens_to_map(outputs.hidden_states[hidden_idx], input_height, input_width)
            features.append(projector(feat, size))
        return features


class DepthEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SwinModel.from_pretrained(str(_require_dir(SWIN_TINY_DIR)))
        self.out_channels = STAGE_CHANNELS

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        # 普通 depth 是单通道，进入 Swin 前 repeat 成 3 通道。
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] == 3:
            pass
        else:
            raise ValueError(f"DepthEncoder expects 1 or 3 channels, got {x.shape[1]}")

        outputs = self.backbone(pixel_values=x, output_hidden_states=True, return_dict=True)
        features = list(outputs.reshaped_hidden_states[:4])
        channels = [feature.shape[1] for feature in features]
        if channels != self.out_channels:
            raise ValueError(f"Expected Swin-T stage channels {self.out_channels}, got {channels}")
        return features
