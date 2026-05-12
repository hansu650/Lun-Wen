"""Main mid-fusion segmentation models."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_lit import BaseLitSeg
from .depth_fft_select import DepthEncoderFFTSelect
from .decoder import ClassContextFPNDecoder, ContextFPNDecoder, SGBRFPNDecoder, SimpleFPNDecoder
from .dformerv2_encoder import DFormerv2_S, load_dformerv2_pretrained
from .encoder import DepthEncoder, RGBEncoder
from .fft_hilo_enhance import FFTHiLoEnhance
from .freq_enhance import FFTFrequencyEnhance
from ..losses import CGPCLoss
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


class LitDFormerV2MidFusion(BaseLitSeg):
    def __init__(
        self,
        num_classes=40,
        lr=1e-4,
        dformerv2_pretrained=None,
        loss_type: str = "ce",
        dice_weight: float = 0.5,
        dgbf_alpha: float = 1.0,
        dgbf_gamma: float = 2.0,
        dgbf_mode: str = "depth_semantic",
        cgpc_weight: float = 0.0,
        cgpc_temperature: float = 0.1,
        cgpc_stage: str = "c3",
        cgpc_min_pixels_per_class: int = 10,
        cgpc_max_pixels_per_class: int = 128,
        cgpc_detach_prototype: bool = True,
    ):
        super().__init__(
            num_classes=num_classes,
            lr=lr,
            loss_type=loss_type,
            dice_weight=dice_weight,
            dgbf_alpha=dgbf_alpha,
            dgbf_gamma=dgbf_gamma,
            dgbf_mode=dgbf_mode,
        )
        self.model = DFormerV2MidFusionSegmentor(
            num_classes=num_classes,
            dformerv2_pretrained=dformerv2_pretrained,
        )
        self.cgpc_weight = float(cgpc_weight)
        if self.cgpc_weight > 0:
            self.cgpc_criterion = CGPCLoss(
                temperature=cgpc_temperature,
                stage=cgpc_stage,
                min_pixels_per_class=cgpc_min_pixels_per_class,
                max_pixels_per_class=cgpc_max_pixels_per_class,
                detach_prototype=cgpc_detach_prototype,
                ignore_index=255,
            )

    def training_step(self, batch, batch_idx):
        if self.cgpc_weight == 0:
            return super().training_step(batch, batch_idx)

        rgb, depth, label = batch["rgb"], batch["depth"], batch["label"]
        label = sanitize_labels(label, num_classes=self.hparams.num_classes, ignore_index=255)

        _, _, fused_feats = self.model.extract_features(rgb, depth)
        logits = self.model.decoder(fused_feats, input_size=rgb.shape[-2:])

        if self.loss_type == "dgbf":
            seg_loss = self.train_criterion(logits, label, depth)
            for name, value in self.train_criterion.last_stats.items():
                self.log(f"train/{name}", value, prog_bar=False, on_step=False, on_epoch=True)
        else:
            seg_loss = self.train_criterion(logits, label)

        cgpc_loss = self.cgpc_criterion(fused_feats, label)
        loss = seg_loss + self.cgpc_weight * cgpc_loss

        self.log("train/seg_loss", seg_loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train/cgpc_loss", cgpc_loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        stats = self.cgpc_criterion.last_stats
        if stats:
            self.log("train/cgpc_num_classes", stats["cgpc_num_classes"], prog_bar=False, on_step=False, on_epoch=True)
            self.log("train/cgpc_num_queries", stats["cgpc_num_queries"], prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)


class DFormerV2ContextDecoderSegmentor(DFormerV2MidFusionSegmentor):
    def __init__(self, num_classes=40, dformerv2_pretrained=None):
        super().__init__(num_classes=num_classes, dformerv2_pretrained=dformerv2_pretrained)
        self.decoder = ContextFPNDecoder(self.rgb_encoder.out_channels, num_classes=num_classes)


class LitDFormerV2ContextDecoder(BaseLitSeg):
    def __init__(
        self,
        num_classes=40,
        lr=1e-4,
        dformerv2_pretrained=None,
        loss_type: str = "ce",
        dice_weight: float = 0.5,
    ):
        super().__init__(num_classes=num_classes, lr=lr, loss_type=loss_type, dice_weight=dice_weight)
        self.model = DFormerV2ContextDecoderSegmentor(
            num_classes=num_classes,
            dformerv2_pretrained=dformerv2_pretrained,
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)


class DFormerV2ClassContextDecoderSegmentor(DFormerV2MidFusionSegmentor):
    def __init__(
        self,
        num_classes=40,
        dformerv2_pretrained=None,
        class_context_channels: int = 64,
        class_context_alpha_init: float = 0.1,
        class_context_alpha_max: float = 0.2,
    ):
        super().__init__(num_classes=num_classes, dformerv2_pretrained=dformerv2_pretrained)
        self.decoder = ClassContextFPNDecoder(
            self.rgb_encoder.out_channels,
            out_channels=128,
            num_classes=num_classes,
            context_channels=class_context_channels,
            alpha_init=class_context_alpha_init,
            alpha_max=class_context_alpha_max,
        )

    def forward(self, rgb, depth, return_aux=False):
        _, _, fused_feats = self.extract_features(rgb, depth)
        return self.decoder(fused_feats, input_size=rgb.shape[-2:], return_aux=return_aux)


class LitDFormerV2ClassContextDecoder(BaseLitSeg):
    def __init__(
        self,
        num_classes=40,
        lr=1e-4,
        dformerv2_pretrained=None,
        loss_type: str = "ce",
        dice_weight: float = 0.5,
        class_context_channels: int = 64,
        class_context_aux_weight: float = 0.2,
        class_context_alpha_init: float = 0.1,
        class_context_alpha_max: float = 0.2,
    ):
        if loss_type != "ce":
            raise ValueError("dformerv2_class_context_decoder only supports --loss_type ce in the first version")
        super().__init__(num_classes=num_classes, lr=lr, loss_type=loss_type, dice_weight=dice_weight)
        self.model = DFormerV2ClassContextDecoderSegmentor(
            num_classes=num_classes,
            dformerv2_pretrained=dformerv2_pretrained,
            class_context_channels=class_context_channels,
            class_context_alpha_init=class_context_alpha_init,
            class_context_alpha_max=class_context_alpha_max,
        )
        self.class_context_aux_weight = float(class_context_aux_weight)

    def training_step(self, batch, batch_idx):
        rgb, depth, label = batch["rgb"], batch["depth"], batch["label"]
        label = sanitize_labels(label, num_classes=self.hparams.num_classes, ignore_index=255)
        final_logits, aux_logits = self.model(rgb, depth, return_aux=True)
        final_loss = self.train_criterion(final_logits, label)
        aux_loss = F.cross_entropy(aux_logits, label, ignore_index=255)
        loss = final_loss + self.class_context_aux_weight * aux_loss

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/final_loss", final_loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train/aux_loss", aux_loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train/context_alpha", self.model.decoder.class_context_block.alpha, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)


class DFormerV2SGBRDecoderSegmentor(DFormerV2MidFusionSegmentor):
    def __init__(
        self,
        num_classes=40,
        dformerv2_pretrained=None,
        sgbr_beta_init: float = 0.05,
        sgbr_beta_max: float = 0.2,
    ):
        super().__init__(num_classes=num_classes, dformerv2_pretrained=dformerv2_pretrained)
        self.decoder = SGBRFPNDecoder(
            self.rgb_encoder.out_channels,
            out_channels=128,
            num_classes=num_classes,
            beta_init=sgbr_beta_init,
            beta_max=sgbr_beta_max,
        )

    def forward(self, rgb, depth, return_aux=False):
        _, _, fused_feats = self.extract_features(rgb, depth)
        return self.decoder(fused_feats, depth=depth, input_size=rgb.shape[-2:], return_aux=return_aux)


class LitDFormerV2SGBRDecoder(BaseLitSeg):
    def __init__(
        self,
        num_classes=40,
        lr=1e-4,
        dformerv2_pretrained=None,
        loss_type: str = "ce",
        dice_weight: float = 0.5,
        sgbr_aux_weight: float = 0.1,
        sgbr_beta_init: float = 0.05,
        sgbr_beta_max: float = 0.2,
    ):
        if loss_type != "ce":
            raise ValueError("dformerv2_sgbr_decoder only supports --loss_type ce in the first version")
        super().__init__(num_classes=num_classes, lr=lr, loss_type=loss_type, dice_weight=dice_weight)
        self.model = DFormerV2SGBRDecoderSegmentor(
            num_classes=num_classes,
            dformerv2_pretrained=dformerv2_pretrained,
            sgbr_beta_init=sgbr_beta_init,
            sgbr_beta_max=sgbr_beta_max,
        )
        self.sgbr_aux_weight = float(sgbr_aux_weight)

    def training_step(self, batch, batch_idx):
        rgb, depth, label = batch["rgb"], batch["depth"], batch["label"]
        label = sanitize_labels(label, num_classes=self.hparams.num_classes, ignore_index=255)
        final_logits, aux_logits = self.model(rgb, depth, return_aux=True)
        final_loss = self.train_criterion(final_logits, label)
        aux_loss = F.cross_entropy(aux_logits, label, ignore_index=255)
        loss = final_loss + self.sgbr_aux_weight * aux_loss

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/final_loss", final_loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train/aux_loss", aux_loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train/sgbr_beta", self.model.decoder.sgbr_block.beta, prog_bar=False, on_step=False, on_epoch=True)
        return loss

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
    def __init__(
        self,
        num_classes=40,
        lr=1e-4,
        dformerv2_pretrained=None,
        cutoff_ratio=0.30,
        loss_type: str = "ce",
        dice_weight: float = 0.5,
    ):
        super().__init__(num_classes=num_classes, lr=lr, loss_type=loss_type, dice_weight=dice_weight)
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
        loss_type: str = "ce",
        dice_weight: float = 0.5,
    ):
        super().__init__(num_classes=num_classes, lr=lr, loss_type=loss_type, dice_weight=dice_weight)
        self.model = DFormerV2FFTFreqEnhanceSegmentor(
            num_classes=num_classes,
            dformerv2_pretrained=dformerv2_pretrained,
            cutoff_ratio=cutoff_ratio,
            gamma_init=gamma_init,
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)


class DFormerV2FFTHiLoEnhanceSegmentor(nn.Module):
    def __init__(
        self,
        num_classes=40,
        dformerv2_pretrained=None,
        cutoff_ratio=0.25,
        alpha_low_init=0.03,
        alpha_high_init=0.10,
        alpha_max=0.5,
        stage_weights=(1.0, 1.0, 1.0, 1.0),
    ):
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
        self.primary_hilo = nn.ModuleList([
            FFTHiLoEnhance(
                ch,
                cutoff_ratio=cutoff_ratio,
                alpha_low_init=alpha_low_init,
                alpha_high_init=alpha_high_init,
                alpha_max=alpha_max,
            )
            for ch in self.rgb_encoder.out_channels
        ])
        self.depth_hilo = nn.ModuleList([
            FFTHiLoEnhance(
                ch,
                cutoff_ratio=cutoff_ratio,
                alpha_low_init=alpha_low_init,
                alpha_high_init=alpha_high_init,
                alpha_max=alpha_max,
            )
            for ch in self.depth_encoder.out_channels
        ])
        self.stage_weights = tuple(float(weight) for weight in stage_weights)

    def extract_features(self, rgb, depth):
        dformer_feats = self.rgb_encoder(rgb, depth)
        depth_feats = self.depth_encoder(depth)

        aligned_depth = []
        for rf, df in zip(dformer_feats, depth_feats):
            if rf.shape[-2:] != df.shape[-2:]:
                df = F.interpolate(df, size=rf.shape[-2:], mode="bilinear", align_corners=False)
            aligned_depth.append(df)

        enhanced_primary = []
        enhanced_depth = []
        for stage_weight, primary_hilo, depth_hilo, primary_feat, depth_feat in zip(
            self.stage_weights,
            self.primary_hilo,
            self.depth_hilo,
            dformer_feats,
            aligned_depth,
        ):
            primary_out = primary_hilo(primary_feat)
            depth_out = depth_hilo(depth_feat)
            enhanced_primary.append(primary_feat + stage_weight * (primary_out - primary_feat))
            enhanced_depth.append(depth_feat + stage_weight * (depth_out - depth_feat))

        fused_feats = [
            fusion(primary_feat, depth_feat)
            for fusion, primary_feat, depth_feat in zip(self.fusions, enhanced_primary, enhanced_depth)
        ]
        return enhanced_primary, enhanced_depth, fused_feats

    def forward(self, rgb, depth):
        _, _, fused_feats = self.extract_features(rgb, depth)
        return self.decoder(fused_feats, input_size=rgb.shape[-2:])


class LitDFormerV2FFTHiLoEnhance(BaseLitSeg):
    def __init__(
        self,
        num_classes=40,
        lr=1e-4,
        dformerv2_pretrained=None,
        cutoff_ratio=0.25,
        alpha_low_init=0.03,
        alpha_high_init=0.10,
        alpha_max=0.5,
        stage_weights=(1.0, 1.0, 1.0, 1.0),
        loss_type: str = "ce",
        dice_weight: float = 0.5,
    ):
        super().__init__(num_classes=num_classes, lr=lr, loss_type=loss_type, dice_weight=dice_weight)
        self.model = DFormerV2FFTHiLoEnhanceSegmentor(
            num_classes=num_classes,
            dformerv2_pretrained=dformerv2_pretrained,
            cutoff_ratio=cutoff_ratio,
            alpha_low_init=alpha_low_init,
            alpha_high_init=alpha_high_init,
            alpha_max=alpha_max,
            stage_weights=stage_weights,
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
