"""TGGA adapter experiment kept active for repeat runs."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_lit import BaseLitSeg
from .mid_fusion import DFormerV2MidFusionSegmentor
from ..utils.metrics import sanitize_labels


def choose_num_groups(channels: int):
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class TGGABlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_classes: int = 40,
        beta_max: float = 0.1,
        beta_init: float = 0.02,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        self.beta_max = float(beta_max)
        raw_beta = math.atanh(float(beta_init) / self.beta_max)
        self.raw_beta = nn.Parameter(torch.tensor(float(raw_beta)))
        self.aux_head = nn.Conv2d(channels, num_classes, kernel_size=1)
        self.gate = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.gate[2].weight)
        nn.init.constant_(self.gate[2].bias, -2.0)

        self.residual = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(choose_num_groups(channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)
        self.forward_calls = 0

    @property
    def effective_beta(self):
        return self.beta_max * torch.tanh(self.raw_beta)

    def _sobel_magnitude(self, x):
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"TGGABlock expects a [B,1,H,W] map for Sobel, got {tuple(x.shape)}")
        sobel_x = self.sobel_x.to(device=x.device, dtype=x.dtype)
        sobel_y = self.sobel_y.to(device=x.device, dtype=x.dtype)
        gx = F.conv2d(x, sobel_x, padding=1)
        gy = F.conv2d(x, sobel_y, padding=1)
        return torch.sqrt(gx.square() + gy.square() + self.eps)

    def _zscore_edge(self, edge):
        mean = edge.mean(dim=(2, 3), keepdim=True)
        std = edge.std(dim=(2, 3), keepdim=True, unbiased=False)
        return ((edge - mean) / (std + self.eps)).clamp(-3.0, 3.0)

    def forward(self, x, rgb, depth):
        self.forward_calls += 1
        b, _, h, w = x.shape
        aux_logits = self.aux_head(x)

        prob = F.softmax(aux_logits.detach(), dim=1)
        conf = prob.max(dim=1, keepdim=True).values
        uncertainty = 1.0 - conf
        semantic_edge = self._zscore_edge(self._sobel_magnitude(conf))

        rgb_i = F.interpolate(rgb, size=(h, w), mode="bilinear", align_corners=False)
        depth_i = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=False)
        rgb_gray = 0.299 * rgb_i[:, 0:1] + 0.587 * rgb_i[:, 1:2] + 0.114 * rgb_i[:, 2:3]

        rgb_edge_raw = self._sobel_magnitude(rgb_gray)
        depth_edge_raw = self._sobel_magnitude(depth_i)
        diff_edge_raw = torch.abs(rgb_edge_raw - depth_edge_raw)
        rgb_edge = self._zscore_edge(rgb_edge_raw)
        depth_edge = self._zscore_edge(depth_edge_raw)
        diff_edge = self._zscore_edge(diff_edge_raw)

        gate_input = torch.cat([uncertainty, semantic_edge, rgb_edge, depth_edge, diff_edge], dim=1)
        if gate_input.shape != (b, 5, h, w):
            raise ValueError(f"TGGA gate input shape mismatch: expected {(b, 5, h, w)}, got {tuple(gate_input.shape)}")

        gate = self.gate(gate_input)
        residual = self.residual(x)
        beta = self.effective_beta
        x_refined = x + beta * gate * residual
        if x_refined.shape != x.shape:
            raise ValueError(f"TGGA output shape mismatch: expected {tuple(x.shape)}, got {tuple(x_refined.shape)}")

        stats = {
            "tgga_beta": beta.detach(),
            "tgga_gate_mean": gate.mean().detach(),
            "tgga_gate_std": gate.std().detach(),
        }
        return x_refined, aux_logits, stats


class DFormerV2TGGAC34Beta002Aux003DetachSemSimpleFPNV2Segmentor(DFormerV2MidFusionSegmentor):
    def __init__(self, num_classes=40, dformerv2_pretrained=None):
        super().__init__(num_classes=num_classes, dformerv2_pretrained=dformerv2_pretrained)
        channels = self.rgb_encoder.out_channels
        self.tgga_c3 = TGGABlock(channels[2], num_classes=num_classes, beta_init=0.02, beta_max=0.1)
        self.tgga_c4 = TGGABlock(channels[3], num_classes=num_classes, beta_init=0.02, beta_max=0.1)

    def extract_features(self, rgb, depth, return_aux=False):
        dformer_feats = self.rgb_encoder(rgb, depth)
        if len(dformer_feats) != 4:
            raise ValueError(f"TGGA expects 4 DFormerV2 feature stages, got {len(dformer_feats)}")
        c1, c2, c3, c4 = dformer_feats
        c3_refined, aux_logits_c3, stats_c3 = self.tgga_c3(c3, rgb, depth)
        c4_refined, aux_logits_c4, stats_c4 = self.tgga_c4(c4, rgb, depth)
        if c3_refined.shape != c3.shape:
            raise ValueError(f"TGGA c3 shape mismatch: {tuple(c3_refined.shape)} vs {tuple(c3.shape)}")
        if c4_refined.shape != c4.shape:
            raise ValueError(f"TGGA c4 shape mismatch: {tuple(c4_refined.shape)} vs {tuple(c4.shape)}")
        refined_feats = [c1, c2, c3_refined, c4_refined]

        depth_feats = self.depth_encoder(depth)
        aligned_depth = []
        for rf, df in zip(refined_feats, depth_feats):
            if rf.shape[-2:] != df.shape[-2:]:
                df = F.interpolate(df, size=rf.shape[-2:], mode="bilinear", align_corners=False)
            aligned_depth.append(df)

        fused_feats = [fusion(r, d) for fusion, r, d in zip(self.fusions, refined_feats, aligned_depth)]
        if not return_aux:
            return refined_feats, aligned_depth, fused_feats
        aux = {
            "aux_logits_c3": aux_logits_c3,
            "aux_logits_c4": aux_logits_c4,
            "tgga_stats_c3": stats_c3,
            "tgga_stats_c4": stats_c4,
            "c3_shape_before": tuple(c3.shape),
            "c3_shape_after": tuple(c3_refined.shape),
            "c4_shape_before": tuple(c4.shape),
            "c4_shape_after": tuple(c4_refined.shape),
        }
        return refined_feats, aligned_depth, fused_feats, aux

    def forward(self, rgb, depth, return_aux=False):
        if not return_aux:
            _, _, fused_feats = self.extract_features(rgb, depth)
            return self.decoder(fused_feats, input_size=rgb.shape[-2:])
        _, _, fused_feats, aux = self.extract_features(rgb, depth, return_aux=True)
        final_logits = self.decoder(fused_feats, input_size=rgb.shape[-2:])
        return final_logits, aux


class LitDFormerV2TGGAC34Beta002Aux003DetachSemSimpleFPNV2(BaseLitSeg):
    def __init__(
        self,
        num_classes=40,
        lr=1e-4,
        dformerv2_pretrained=None,
        loss_type: str = "ce",
        dice_weight: float = 0.5,
    ):
        if loss_type != "ce":
            raise ValueError(
                "dformerv2_tgga_c34_beta002_aux003_detachsem_simplefpn_v2 only supports --loss_type ce"
            )
        super().__init__(num_classes=num_classes, lr=lr, loss_type=loss_type, dice_weight=dice_weight)
        self.model = DFormerV2TGGAC34Beta002Aux003DetachSemSimpleFPNV2Segmentor(
            num_classes=num_classes,
            dformerv2_pretrained=dformerv2_pretrained,
        )
        self.aux_weight_c3 = 0.03
        self.aux_weight_c4 = 0.03

    def training_step(self, batch, batch_idx):
        rgb, depth, label = batch["rgb"], batch["depth"], batch["label"]
        label = sanitize_labels(label, num_classes=self.hparams.num_classes, ignore_index=255)
        final_logits, aux = self.model(rgb, depth, return_aux=True)
        loss_main = self.train_criterion(final_logits, label)

        aux_logits_c3 = F.interpolate(
            aux["aux_logits_c3"],
            size=label.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        aux_logits_c4 = F.interpolate(
            aux["aux_logits_c4"],
            size=label.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        loss_aux_c3 = F.cross_entropy(aux_logits_c3, label, ignore_index=255)
        loss_aux_c4 = F.cross_entropy(aux_logits_c4, label, ignore_index=255)
        loss = loss_main + self.aux_weight_c3 * loss_aux_c3 + self.aux_weight_c4 * loss_aux_c4

        stats_c3 = aux["tgga_stats_c3"]
        stats_c4 = aux["tgga_stats_c4"]
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/main_loss", loss_main, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train/tgga_aux_loss_c3", loss_aux_c3, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train/tgga_aux_loss_c4", loss_aux_c4, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train/tgga_beta_c3", stats_c3["tgga_beta"], prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/tgga_beta_c4", stats_c4["tgga_beta"], prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/tgga_gate_c3_mean", stats_c3["tgga_gate_mean"], prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/tgga_gate_c4_mean", stats_c4["tgga_gate_mean"], prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/tgga_gate_c3_std", stats_c3["tgga_gate_std"], prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/tgga_gate_c4_std", stats_c4["tgga_gate_std"], prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
