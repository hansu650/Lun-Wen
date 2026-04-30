from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18
from transformers import Dinov2Model, SwinModel


ROOT = Path(__file__).resolve().parents[3]
DINOV2_SMALL_DIR = ROOT / "pretrained" / "dinov2_small"
SWIN_TINY_DIR = ROOT / "pretrained" / "swin_tiny"
STAGE_CHANNELS = [96, 192, 384, 768]
HIDDEN_STATE_INDICES = (3, 6, 9, 12)


class RGBEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # RGB branch: replace teacher ResNet18 with DINOv2-small.
        self.backbone = Dinov2Model.from_pretrained(str(DINOV2_SMALL_DIR))
        self.patch_size = int(self.backbone.config.patch_size)
        self.out_channels = STAGE_CHANNELS
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.backbone.config.hidden_size, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            )
            for out_channels in self.out_channels
        ])

    def forward(self, x, return_cls: bool = False):
        b, _, h, w = x.shape
        gh = h // self.patch_size
        gw = w // self.patch_size
        sizes = [(h // 4, w // 4), (h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)]
        outputs = self.backbone(pixel_values=x, output_hidden_states=True, return_dict=True)

        feats = []
        cls_tokens = []
        for projector, hidden_idx, size in zip(self.projectors, HIDDEN_STATE_INDICES, sizes):
            hidden_state = outputs.hidden_states[hidden_idx]
            # DINOv2 第 0 个 token 是 CLS，用作 RGB 全局语义指导。
            cls_tokens.append(hidden_state[:, 0, :])
            tokens = hidden_state[:, 1:, :]
            x = tokens.transpose(1, 2).reshape(b, self.backbone.config.hidden_size, gh, gw)
            x = projector(x)
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
            feats.append(x)
        if return_cls:
            return feats, cls_tokens
        return feats


class DepthEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Depth branch: replace teacher ResNet18 with Swin-Tiny.
        self.backbone = SwinModel.from_pretrained(str(SWIN_TINY_DIR))
        self.out_channels = STAGE_CHANNELS

    def forward(self, x):
        # 单通道 metric depth repeat 成 3 通道后送入 Swin-T。
        x = x.repeat(1, 3, 1, 1)
        outputs = self.backbone(pixel_values=x, output_hidden_states=True, return_dict=True)
        feats = list(outputs.reshaped_hidden_states[:4])
        return feats


class EarlyFusionEncoder(nn.Module):
    def __init__(self):
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

    def forward(self, x):
        feats = []
        x = self.layer0(x)
        x = self.layer1(x); feats.append(x)
        x = self.layer2(x); feats.append(x)
        x = self.layer3(x); feats.append(x)
        x = self.layer4(x); feats.append(x)
        return feats
