"""Encoder modules for RGB, depth, and early-fusion variants."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model, SwinModel
from torchvision.models import ResNet18_Weights, resnet18


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PRETRAINED_ROOT = PROJECT_ROOT / "pretrained"
DINOV2_SMALL_DIR = PRETRAINED_ROOT / "dinov2_small"
SWIN_TINY_DIR = PRETRAINED_ROOT / "swin_tiny"

# 当前改进方向：
# 1. RGB 分支换成 DINO，用更强的预训练语义特征处理彩色图。
# 2. depth 分支换成 Swin Transformer，用分层结构处理几何信息。
# 3. backbone 先走小模型路线：DINOv2-S + Swin-T。
# 这里固定 Swin-T 的四层通道数，后面的 fusion/decoder 都跟着这组维度走。
SWIN_T_STAGE_CHANNELS = [96, 192, 384, 768]


def _require_local_model_dir(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Local pretrained directory not found: {path}. "
            "Please place the Hugging Face checkpoint under the expected "
            "pretrained/ folder before training."
        )
    return path


def _compute_pyramid_sizes(height: int, width: int) -> list[tuple[int, int]]:
    # DINO 输出的是 token，不是天然的 feature pyramid。
    # 这里手动指定四个目标尺度，让 DINO 的四层 token 特征对齐到 Swin-T 的 c1-c4。
    return [
        (max(1, height // 4), max(1, width // 4)),
        (max(1, height // 8), max(1, width // 8)),
        (max(1, height // 16), max(1, width // 16)),
        (max(1, height // 32), max(1, width // 32)),
    ]


class DinoStageNeck(nn.Module):
    """Project one DINO hidden state into one Swin-T-sized stage map."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # DINOv2-S 的 token 通道是 384。
        # c1-c4 需要分别变成 [96, 192, 384, 768]，所以每层用 1x1 conv 做投影。
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


class RGBEncoder(nn.Module):
    """RGB branch: DINOv2 token features projected to Swin-T stage widths."""

    def __init__(
        self,
        model_dir: Path | None = None,
        hidden_state_indices: tuple[int, int, int, int] = (3, 6, 9, 12),
    ) -> None:
        super().__init__()
        # RGB 分支默认读取 DINOv2-S 权重。
        # 权重目录固定为 pretrained/dinov2_small。
        model_dir = _require_local_model_dir(model_dir or DINOV2_SMALL_DIR)
        self.backbone = Dinov2Model.from_pretrained(str(model_dir))
        self.backbone.train()
        # 从 DINO 的不同 transformer 层取特征，构造成四个 stage。
        self.hidden_state_indices = hidden_state_indices
        self.patch_size = int(self.backbone.config.patch_size)
        self.out_channels = SWIN_T_STAGE_CHANNELS
        self.stage_necks = nn.ModuleList(
            [
                DinoStageNeck(self.backbone.config.hidden_size, out_channels)
                for out_channels in self.out_channels
            ]
        )

    def _tokens_to_map(
        self,
        tokens: torch.Tensor,
        input_height: int,
        input_width: int,
    ) -> torch.Tensor:
        # DINO 输出格式是 (B, token_count, C)。
        # 第一个 token 通常是 CLS token，分割任务里先去掉，再 reshape 回二维特征图。
        batch_size, token_count, channels = tokens.shape
        grid_h = max(1, input_height // self.patch_size)
        grid_w = max(1, input_width // self.patch_size)

        if token_count == grid_h * grid_w + 1:
            tokens = tokens[:, 1:, :]
            token_count = tokens.shape[1]

        if token_count != grid_h * grid_w:
            raise ValueError(
                f"DINO token count {token_count} does not match grid {grid_h}x{grid_w}."
            )

        return tokens.transpose(1, 2).reshape(batch_size, channels, grid_h, grid_w)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        input_height, input_width = x.shape[-2:]
        outputs = self.backbone(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True,
        )

        target_sizes = _compute_pyramid_sizes(input_height, input_width)
        features: list[torch.Tensor] = []

        # hidden_states[3/6/9/12] -> token map -> 1x1 投影 -> resize 到 c1-c4 尺度。
        for neck, hidden_idx, target_size in zip(
            self.stage_necks, self.hidden_state_indices, target_sizes
        ):
            hidden = outputs.hidden_states[hidden_idx]
            feature_map = self._tokens_to_map(hidden, input_height, input_width)
            features.append(neck(feature_map, target_size))

        return features


class DepthEncoder(nn.Module):
    """Depth branch: Swin-T over repeated 1-channel depth input."""

    def __init__(self, model_dir: Path | None = None) -> None:
        super().__init__()
        # depth 分支默认读取 pretrained/swin_tiny。
        # 这个目录应放 microsoft/swin-tiny-patch4-window7-224。
        model_dir = _require_local_model_dir(model_dir or SWIN_TINY_DIR)
        self.backbone = SwinModel.from_pretrained(str(model_dir))
        self.backbone.train()
        self.out_channels = SWIN_T_STAGE_CHANNELS

    def _repeat_depth_to_rgb(self, x: torch.Tensor) -> torch.Tensor:
        # Hugging Face 的 Swin 预训练权重是 3 通道输入。
        # NYU depth 是 1 通道，所以这里复制成 3 通道再送入 Swin。
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.shape[1] == 1:
            return x.repeat(1, 3, 1, 1)
        if x.shape[1] == 3:
            return x
        raise ValueError(f"Depth input must have 1 or 3 channels, got {x.shape[1]}.")

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self._repeat_depth_to_rgb(x)
        outputs = self.backbone(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True,
        )
        # Swin-T 天然输出四个分层 feature map，正好作为 depth c1-c4。
        features = list(outputs.reshaped_hidden_states[:4])

        actual_channels = [feature.shape[1] for feature in features]
        if actual_channels != self.out_channels:
            raise ValueError(
                "The loaded depth Swin checkpoint does not look like Swin-T. "
                f"Expected stage channels {self.out_channels}, got {actual_channels}."
            )

        return features


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
