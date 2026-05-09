"""Main mid-fusion segmentation models."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_lit import BaseLitSeg
from .contrastive_loss import CrossModalInfoNCELoss
from .depth_fft_select import DepthEncoderFFTSelect
from .decoder import SimpleFPNDecoder
from .dformerv2_encoder import DFormerv2_S, load_dformerv2_pretrained
from .encoder import DepthEncoder, RGBEncoder
from .freq_enhance import FFTFrequencyEnhance
from .freq_cov_loss import MultiScaleFrequencyCovarianceLoss
from .mask_reconstruction_loss import FeatureMaskReconstructionLoss
from ..utils.metrics import sanitize_labels


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
    def __init__(self, num_classes=40, lr=1e-4):
        super().__init__(num_classes=num_classes, lr=lr)
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


class LitDFormerV2MidFusion(BaseLitSeg):
    def __init__(self, num_classes=40, lr=1e-4, dformerv2_pretrained=None):
        super().__init__(num_classes=num_classes, lr=lr)
        self.model = DFormerV2MidFusionSegmentor(
            num_classes=num_classes,
            dformerv2_pretrained=dformerv2_pretrained,
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)


class DFormerV2DepthFFTSelectSegmentor(nn.Module):
    def __init__(self, num_classes=40, dformerv2_pretrained=None, cutoff_ratio=0.30):
        super().__init__()
        self.rgb_encoder = DFormerv2_S()
        if dformerv2_pretrained:
            self.pretrained_load_stats = load_dformerv2_pretrained(self.rgb_encoder, dformerv2_pretrained)
        else:
            self.pretrained_load_stats = None
        self.depth_encoder = DepthEncoderFFTSelect(cutoff_ratio=cutoff_ratio)
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


class LitDFormerV2DepthFFTSelect(BaseLitSeg):
    def __init__(self, num_classes=40, lr=1e-4, dformerv2_pretrained=None, cutoff_ratio=0.30):
        super().__init__(num_classes=num_classes, lr=lr)
        self.model = DFormerV2DepthFFTSelectSegmentor(
            num_classes=num_classes,
            dformerv2_pretrained=dformerv2_pretrained,
            cutoff_ratio=cutoff_ratio,
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)


class DFormerV2FFTFreqEnhanceSegmentor(DFormerV2MidFusionSegmentor):
    def __init__(
        self,
        num_classes=40,
        dformerv2_pretrained=None,
        cutoff_ratio=0.25,
        gamma_init=0.05,
    ):
        super().__init__(num_classes=num_classes, dformerv2_pretrained=dformerv2_pretrained)

        primary_channels = self.rgb_encoder.out_channels
        depth_channels = self.depth_encoder.out_channels
        self.primary_enhance = nn.ModuleList([
            FFTFrequencyEnhance(ch, cutoff_ratio=cutoff_ratio, gamma_init=gamma_init)
            for ch in primary_channels
        ])
        self.depth_enhance = nn.ModuleList([
            FFTFrequencyEnhance(ch, cutoff_ratio=cutoff_ratio, gamma_init=gamma_init)
            for ch in depth_channels
        ])

    def extract_features(self, rgb, depth):
        dformer_feats = self.rgb_encoder(rgb, depth)
        depth_feats = self.depth_encoder(depth)

        aligned_depth = []
        for rf, df in zip(dformer_feats, depth_feats):
            if rf.shape[-2:] != df.shape[-2:]:
                df = F.interpolate(df, size=rf.shape[-2:], mode="bilinear", align_corners=False)
            aligned_depth.append(df)

        enhanced_primary = [
            enhance(primary_feat)
            for enhance, primary_feat in zip(self.primary_enhance, dformer_feats)
        ]
        enhanced_depth = [
            enhance(depth_feat)
            for enhance, depth_feat in zip(self.depth_enhance, aligned_depth)
        ]
        fused_feats = [
            fusion(primary_feat, depth_feat)
            for fusion, primary_feat, depth_feat in zip(self.fusions, enhanced_primary, enhanced_depth)
        ]
        return enhanced_primary, enhanced_depth, fused_feats


class LitDFormerV2FFTFreqEnhance(BaseLitSeg):
    def __init__(
        self,
        num_classes=40,
        lr=1e-4,
        dformerv2_pretrained=None,
        cutoff_ratio=0.25,
        gamma_init=0.05,
    ):
        super().__init__(num_classes=num_classes, lr=lr)
        self.model = DFormerV2FFTFreqEnhanceSegmentor(
            num_classes=num_classes,
            dformerv2_pretrained=dformerv2_pretrained,
            cutoff_ratio=cutoff_ratio,
            gamma_init=gamma_init,
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)


class LitDFormerV2MSFreqCov(BaseLitSeg):
    def __init__(
        self,
        num_classes=40,
        lr=1e-4,
        dformerv2_pretrained=None,
        lambda_freq=0.01,
        freq_eta=1.0,
        freq_proj_dim=64,
        freq_kernel_size=3,
        freq_stage_weights=(1.0, 1.0, 1.0, 1.0),
    ):
        super().__init__(num_classes=num_classes, lr=lr)
        self.lambda_freq = float(lambda_freq)
        self.model = DFormerV2MidFusionSegmentor(
            num_classes=num_classes,
            dformerv2_pretrained=dformerv2_pretrained,
        )
        self.freqcov_loss = MultiScaleFrequencyCovarianceLoss(
            rgb_channels=self.model.rgb_encoder.out_channels,
            depth_channels=self.model.depth_encoder.out_channels,
            proj_dim=freq_proj_dim,
            kernel_size=freq_kernel_size,
            eta=freq_eta,
            stage_weights=freq_stage_weights,
        )

    def training_step(self, batch, batch_idx):
        rgb, depth, label = batch["rgb"], batch["depth"], batch["label"]
        label = sanitize_labels(label, num_classes=self.hparams.num_classes, ignore_index=255)

        dformer_feats, aligned_depth, fused_feats = self.model.extract_features(rgb, depth)
        logits = self.model.decoder(fused_feats, input_size=rgb.shape[-2:])
        seg_loss = self.criterion(logits, label)
        freq_loss, freq_dict = self.freqcov_loss(dformer_feats, aligned_depth)
        loss = seg_loss + self.lambda_freq * freq_loss

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/seg_loss", seg_loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train/freqcov_loss", freq_loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train/lambda_freq", self.lambda_freq, prog_bar=False, on_step=False, on_epoch=True)
        for name, value in freq_dict.items():
            self.log(name, value, prog_bar=False, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)


class LitDFormerV2FeatMaskRecC34(BaseLitSeg):
    def __init__(
        self,
        num_classes=40,
        lr=1e-4,
        dformerv2_pretrained=None,
        lambda_mask=0.01,
        mask_ratio_depth=0.30,
        mask_ratio_primary=0.15,
        maskrec_alpha=0.5,
        maskrec_loss_type="smooth_l1",
        maskrec_stage_weights=(1.0, 1.0, 1.0, 1.0),
    ):
        super().__init__(num_classes=num_classes, lr=lr)
        self.lambda_mask = float(lambda_mask)
        self.model = DFormerV2MidFusionSegmentor(
            num_classes=num_classes,
            dformerv2_pretrained=dformerv2_pretrained,
        )
        self.maskrec_loss = FeatureMaskReconstructionLoss(
            primary_channels=self.model.rgb_encoder.out_channels,
            depth_channels=self.model.depth_encoder.out_channels,
            stage_weights=maskrec_stage_weights,
            mask_ratio_depth=mask_ratio_depth,
            mask_ratio_primary=mask_ratio_primary,
            alpha=maskrec_alpha,
            loss_type=maskrec_loss_type,
        )

    def training_step(self, batch, batch_idx):
        rgb, depth, label = batch["rgb"], batch["depth"], batch["label"]
        label = sanitize_labels(label, num_classes=self.hparams.num_classes, ignore_index=255)

        dformer_feats, aligned_depth, fused_feats = self.model.extract_features(rgb, depth)
        logits = self.model.decoder(fused_feats, input_size=rgb.shape[-2:])
        seg_loss = self.criterion(logits, label)
        maskrec_loss, maskrec_dict = self.maskrec_loss(dformer_feats, aligned_depth)
        loss = seg_loss + self.lambda_mask * maskrec_loss

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/seg_loss", seg_loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train/maskrec_loss", maskrec_loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train/lambda_mask", self.lambda_mask, prog_bar=False, on_step=False, on_epoch=True)
        for name, value in maskrec_dict.items():
            self.log(name, value, prog_bar=False, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)


class LitDFormerV2CMInfoNCE(BaseLitSeg):
    def __init__(
        self,
        num_classes=40,
        lr=1e-4,
        dformerv2_pretrained=None,
        lambda_contrast=0.005,
        temperature=0.1,
        proj_dim=64,
        sample_points=256,
        stage_weights=(0.0, 0.0, 1.0, 1.0),
    ):
        super().__init__(num_classes=num_classes, lr=lr)
        self.lambda_contrast = float(lambda_contrast)
        self.model = DFormerV2MidFusionSegmentor(
            num_classes=num_classes,
            dformerv2_pretrained=dformerv2_pretrained,
        )
        self.contrast_loss = CrossModalInfoNCELoss(
            primary_channels=self.model.rgb_encoder.out_channels,
            depth_channels=self.model.depth_encoder.out_channels,
            proj_dim=proj_dim,
            temperature=temperature,
            sample_points=sample_points,
            stage_weights=stage_weights,
        )

    def training_step(self, batch, batch_idx):
        rgb, depth, label = batch["rgb"], batch["depth"], batch["label"]
        label = sanitize_labels(label, num_classes=self.hparams.num_classes, ignore_index=255)

        dformer_feats, aligned_depth, fused_feats = self.model.extract_features(rgb, depth)
        logits = self.model.decoder(fused_feats, input_size=rgb.shape[-2:])
        seg_loss = self.criterion(logits, label)
        contrast_loss, contrast_dict = self.contrast_loss(dformer_feats, aligned_depth)
        loss = seg_loss + self.lambda_contrast * contrast_loss

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/seg_loss", seg_loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train/contrast_loss", contrast_loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train/lambda_contrast", self.lambda_contrast, prog_bar=False, on_step=False, on_epoch=True)
        for name, value in contrast_dict.items():
            self.log(name, value, prog_bar=False, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
