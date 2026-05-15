"""Active mid-fusion segmentation models."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_lit import BaseLitSeg
from .decoder import OfficialHamDecoder, SimpleFPNDecoder
from .dformerv2_encoder import DFormerv2_S, load_dformerv2_pretrained
from .encoder import DepthEncoder, RGBEncoder


class GatedFusion(nn.Module):
    def __init__(self, rgb_channels, depth_channels):
        super().__init__()
        self.depth_proj = nn.Conv2d(depth_channels, rgb_channels, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(rgb_channels * 2, rgb_channels, 1, bias=False),
            nn.BatchNorm2d(rgb_channels),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(rgb_channels, rgb_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(rgb_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb_feat, depth_feat):
        d = self.depth_proj(depth_feat)
        g = self.gate(torch.cat([rgb_feat, d], dim=1))
        fused = g * rgb_feat + (1 - g) * d
        return self.refine(fused)


class MidFusionSegmentor(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()
        self.rgb_encoder = RGBEncoder()
        self.depth_encoder = DepthEncoder()
        self.fusions = nn.ModuleList([
            GatedFusion(rgb_ch, depth_ch)
            for rgb_ch, depth_ch in zip(self.rgb_encoder.out_channels, self.depth_encoder.out_channels)
        ])
        self.decoder = SimpleFPNDecoder(self.rgb_encoder.out_channels, num_classes=num_classes)

    def forward(self, rgb, depth):
        rgb_feats = self.rgb_encoder(rgb)
        depth_feats = self.depth_encoder(depth)

        aligned_depth = []
        for rf, df in zip(rgb_feats, depth_feats):
            if rf.shape[-2:] != df.shape[-2:]:
                df = F.interpolate(df, size=rf.shape[-2:], mode="bilinear", align_corners=False)
            aligned_depth.append(df)

        fused_feats = [f(rf, df) for f, rf, df in zip(self.fusions, rgb_feats, aligned_depth)]
        return self.decoder(fused_feats, input_size=rgb.shape[-2:])


class LitMidFusion(BaseLitSeg):
    def __init__(self, num_classes=40, lr=1e-4, loss_type: str = "ce", dice_weight: float = 0.5):
        super().__init__(num_classes=num_classes, lr=lr, loss_type=loss_type, dice_weight=dice_weight)
        self.model = MidFusionSegmentor(num_classes=num_classes)


class DFormerV2MidFusionSegmentor(nn.Module):
    def __init__(self, num_classes=40, dformerv2_pretrained=None):
        super().__init__()
        self.rgb_encoder = DFormerv2_S()
        if dformerv2_pretrained:
            self.pretrained_load_stats = load_dformerv2_pretrained(self.rgb_encoder, dformerv2_pretrained)
        else:
            self.pretrained_load_stats = None
        self.depth_encoder = DepthEncoder()
        self.fusions = nn.ModuleList([
            GatedFusion(rgb_ch, depth_ch)
            for rgb_ch, depth_ch in zip(self.rgb_encoder.out_channels, self.depth_encoder.out_channels)
        ])
        self.decoder = SimpleFPNDecoder(self.rgb_encoder.out_channels, num_classes=num_classes)

    def extract_features(self, rgb, depth):
        dformer_feats = self.rgb_encoder(rgb, depth)
        depth_feats = self.depth_encoder(depth)

        aligned_depth = []
        for rf, df in zip(dformer_feats, depth_feats):
            if rf.shape[-2:] != df.shape[-2:]:
                df = F.interpolate(df, size=rf.shape[-2:], mode="bilinear", align_corners=False)
            aligned_depth.append(df)

        fused_feats = [fusion(r, d) for fusion, r, d in zip(self.fusions, dformer_feats, aligned_depth)]
        return dformer_feats, aligned_depth, fused_feats

    def forward(self, rgb, depth):
        _, _, fused_feats = self.extract_features(rgb, depth)
        return self.decoder(fused_feats, input_size=rgb.shape[-2:])


class DFormerV2HamDecoderSegmentor(DFormerV2MidFusionSegmentor):
    def __init__(self, num_classes=40, dformerv2_pretrained=None):
        super().__init__(num_classes=num_classes, dformerv2_pretrained=dformerv2_pretrained)
        self.decoder = OfficialHamDecoder(
            self.rgb_encoder.out_channels,
            channels=512,
            num_classes=num_classes,
        )


class LitDFormerV2MidFusion(BaseLitSeg):
    def __init__(
        self,
        num_classes=40,
        lr=1e-4,
        dformerv2_pretrained=None,
        loss_type: str = "ce",
        dice_weight: float = 0.5,
    ):
        super().__init__(
            num_classes=num_classes,
            lr=lr,
            loss_type=loss_type,
            dice_weight=dice_weight,
        )
        self.model = DFormerV2MidFusionSegmentor(
            num_classes=num_classes,
            dformerv2_pretrained=dformerv2_pretrained,
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)

class LitDFormerV2HamDecoder(BaseLitSeg):
    def __init__(
        self,
        num_classes=40,
        lr=1e-4,
        dformerv2_pretrained=None,
        loss_type: str = "ce",
        dice_weight: float = 0.5,
    ):
        super().__init__(
            num_classes=num_classes,
            lr=lr,
            loss_type=loss_type,
            dice_weight=dice_weight,
        )
        self.model = DFormerV2HamDecoderSegmentor(
            num_classes=num_classes,
            dformerv2_pretrained=dformerv2_pretrained,
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
