"""Logit-only PMAD/PrimKD distillation Lightning module."""
import math

import torch
import torch.nn.functional as F

from .base_lit import BaseLitSeg
from .mid_fusion import DFormerV2MidFusionSegmentor
from .teacher_model import DFormerV2GeometryPrimaryTeacherSegmentor
from ..utils.metrics import sanitize_labels


class LitDFormerV2PrimKD(BaseLitSeg):
    def __init__(
        self,
        num_classes=40,
        lr=1e-4,
        dformerv2_pretrained=None,
        teacher_ckpt=None,
        kd_weight=0.2,
        kd_temperature=4.0,
        loss_type: str = "ce",
        dice_weight: float = 0.5,
    ):
        super().__init__(num_classes=num_classes, lr=lr, loss_type=loss_type, dice_weight=dice_weight)
        if teacher_ckpt is None:
            raise ValueError("teacher_ckpt is required for dformerv2_primkd_logit_only")
        self.model = DFormerV2MidFusionSegmentor(
            num_classes=num_classes,
            dformerv2_pretrained=dformerv2_pretrained,
        )
        self.teacher = DFormerV2GeometryPrimaryTeacherSegmentor(num_classes=num_classes)
        state = torch.load(teacher_ckpt, map_location="cpu")
        teacher_state = {
            key[len("model.") :]: value
            for key, value in state.items()
            if key.startswith("model.")
        }
        self.teacher.load_state_dict(teacher_state, strict=True)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.kd_weight = float(kd_weight)
        self.kd_temperature = float(kd_temperature)

    def forward(self, rgb, depth):
        return self.model(rgb, depth)

    @staticmethod
    def segmentation_kl_loss(student_logits, teacher_logits, label, temperature, ignore_index=255):
        valid = label != ignore_index
        student = student_logits.permute(0, 2, 3, 1)[valid]
        teacher = teacher_logits.permute(0, 2, 3, 1)[valid]
        if student.numel() == 0:
            return student_logits.sum() * 0.0
        student_log_prob = F.log_softmax(student / temperature, dim=1)
        teacher_prob = F.softmax(teacher / temperature, dim=1)
        return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (temperature ** 2)

    def training_step(self, batch, batch_idx):
        rgb, depth, label = batch["rgb"], batch["depth"], batch["label"]
        label = sanitize_labels(label, num_classes=self.hparams.num_classes, ignore_index=255)
        student_logits = self(rgb, depth)
        self.teacher.eval()
        with torch.no_grad():
            teacher_logits = self.teacher(rgb, depth)
        ce_loss = self.train_criterion(student_logits, label)
        kd_loss = self.segmentation_kl_loss(
            student_logits,
            teacher_logits,
            label,
            self.kd_temperature,
            ignore_index=255,
        )
        loss = ce_loss + self.kd_weight * kd_loss
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/ce_loss", ce_loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train/kd_loss", kd_loss, prog_bar=False, on_step=True, on_epoch=True)
        return loss

    def export_state_dict(self):
        return self.model.state_dict()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr, weight_decay=0.01)


class LitDFormerV2PrimKDBoundaryConf(LitDFormerV2PrimKD):
    confidence_threshold = 0.40
    confidence_power = 1.5
    boundary_boost = 1.0

    @staticmethod
    def semantic_boundary_mask(label, ignore_index=255):
        valid = label != ignore_index
        boundary = torch.zeros_like(valid, dtype=torch.bool)
        if label.shape[-1] > 1:
            pair_valid = valid[:, :, 1:] & valid[:, :, :-1]
            pair_diff = (label[:, :, 1:] != label[:, :, :-1]) & pair_valid
            boundary[:, :, 1:] |= pair_diff
            boundary[:, :, :-1] |= pair_diff
        if label.shape[-2] > 1:
            pair_valid = valid[:, 1:, :] & valid[:, :-1, :]
            pair_diff = (label[:, 1:, :] != label[:, :-1, :]) & pair_valid
            boundary[:, 1:, :] |= pair_diff
            boundary[:, :-1, :] |= pair_diff
        return boundary & valid

    @classmethod
    def selective_kl_loss(cls, student_logits, teacher_logits, label, temperature, ignore_index=255):
        valid = label != ignore_index
        if valid.sum() == 0:
            zero = student_logits.sum() * 0.0
            stats = {
                "kd_mask_ratio": zero.detach(),
                "kd_boundary_ratio": zero.detach(),
                "kd_conf_mean": zero.detach(),
            }
            return zero, stats

        with torch.no_grad():
            teacher_conf = F.softmax(teacher_logits, dim=1).amax(dim=1)
            boundary = cls.semantic_boundary_mask(label, ignore_index=ignore_index)
            trust = ((teacher_conf >= cls.confidence_threshold) | boundary) & valid
            weight = teacher_conf.pow(cls.confidence_power)
            weight = weight * (1.0 + cls.boundary_boost * boundary.float())
            weight = weight * trust.float()
            mean_weight = weight[valid].mean().clamp_min(1e-6)
            weight = weight / mean_weight

        student = student_logits.permute(0, 2, 3, 1)[valid]
        teacher = teacher_logits.permute(0, 2, 3, 1)[valid]
        pixel_weight = weight[valid]
        if pixel_weight.sum() <= 0:
            zero = student_logits.sum() * 0.0
            stats = {
                "kd_mask_ratio": zero.detach(),
                "kd_boundary_ratio": boundary[valid].float().mean(),
                "kd_conf_mean": teacher_conf[valid].mean(),
            }
            return zero, stats

        student_log_prob = F.log_softmax(student / temperature, dim=1)
        teacher_prob = F.softmax(teacher / temperature, dim=1)
        kl_pixel = F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(dim=1)
        loss = (kl_pixel * pixel_weight).sum() / pixel_weight.sum().clamp_min(1.0)
        loss = loss * (temperature ** 2)
        stats = {
            "kd_mask_ratio": trust[valid].float().mean(),
            "kd_boundary_ratio": boundary[valid].float().mean(),
            "kd_conf_mean": teacher_conf[valid].mean(),
        }
        return loss, stats

    def training_step(self, batch, batch_idx):
        rgb, depth, label = batch["rgb"], batch["depth"], batch["label"]
        label = sanitize_labels(label, num_classes=self.hparams.num_classes, ignore_index=255)
        student_logits = self(rgb, depth)
        self.teacher.eval()
        with torch.no_grad():
            teacher_logits = self.teacher(rgb, depth)
        ce_loss = self.train_criterion(student_logits, label)
        kd_loss, kd_stats = self.selective_kl_loss(
            student_logits,
            teacher_logits,
            label,
            self.kd_temperature,
            ignore_index=255,
        )
        loss = ce_loss + self.kd_weight * kd_loss
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/ce_loss", ce_loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train/kd_loss", kd_loss, prog_bar=False, on_step=True, on_epoch=True)
        for name, value in kd_stats.items():
            self.log(f"train/{name}", value, prog_bar=False, on_step=True, on_epoch=True)
        return loss


class LitDFormerV2PrimKDCorrectEntropy(LitDFormerV2PrimKD):
    def __init__(
        self,
        num_classes=40,
        lr=1e-4,
        dformerv2_pretrained=None,
        teacher_ckpt=None,
        kd_weight=0.2,
        kd_temperature=4.0,
        kd_entropy_threshold=0.25,
        loss_type: str = "ce",
        dice_weight: float = 0.5,
    ):
        super().__init__(
            num_classes=num_classes,
            lr=lr,
            dformerv2_pretrained=dformerv2_pretrained,
            teacher_ckpt=teacher_ckpt,
            kd_weight=kd_weight,
            kd_temperature=kd_temperature,
            loss_type=loss_type,
            dice_weight=dice_weight,
        )
        self.kd_entropy_threshold = float(kd_entropy_threshold)

    @staticmethod
    def correct_entropy_kl_loss(student_logits, teacher_logits, label, temperature, entropy_threshold, ignore_index=255):
        valid = label != ignore_index
        if valid.sum() == 0:
            zero = student_logits.sum() * 0.0
            stats = {
                "kd_mask_ratio": zero.detach(),
                "kd_entropy_mean": zero.detach(),
                "kd_entropy_selected_mean": zero.detach(),
                "kd_teacher_valid_acc": zero.detach(),
                "kd_teacher_selected_acc": zero.detach(),
                "kd_selected_kl": zero.detach(),
            }
            return zero, stats

        with torch.no_grad():
            teacher_prob_raw = F.softmax(teacher_logits, dim=1)
            entropy = -(teacher_prob_raw * torch.log(teacher_prob_raw.clamp_min(1e-8))).sum(dim=1)
            entropy = entropy / math.log(float(teacher_prob_raw.shape[1]))
            teacher_pred = teacher_prob_raw.argmax(dim=1)
            teacher_correct = teacher_pred == label
            selected = valid & teacher_correct & (entropy <= entropy_threshold)

        if selected.sum() == 0:
            zero = student_logits.sum() * 0.0
            stats = {
                "kd_mask_ratio": zero.detach(),
                "kd_entropy_mean": entropy[valid].mean(),
                "kd_entropy_selected_mean": zero.detach(),
                "kd_teacher_valid_acc": teacher_correct[valid].float().mean(),
                "kd_teacher_selected_acc": zero.detach(),
                "kd_selected_kl": zero.detach(),
            }
            return zero, stats

        student = student_logits.permute(0, 2, 3, 1)[selected]
        teacher = teacher_logits.permute(0, 2, 3, 1)[selected]
        student_log_prob = F.log_softmax(student / temperature, dim=1)
        teacher_prob = F.softmax(teacher / temperature, dim=1)
        kl_pixel = F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(dim=1)
        valid_count = valid.float().sum().clamp_min(1.0)
        loss = kl_pixel.sum() / valid_count
        loss = loss * (temperature ** 2)
        stats = {
            "kd_mask_ratio": selected[valid].float().mean(),
            "kd_entropy_mean": entropy[valid].mean(),
            "kd_entropy_selected_mean": entropy[selected].mean(),
            "kd_teacher_valid_acc": teacher_correct[valid].float().mean(),
            "kd_teacher_selected_acc": teacher_correct[selected].float().mean(),
            "kd_selected_kl": kl_pixel.mean().detach() * (temperature ** 2),
        }
        return loss, stats

    def training_step(self, batch, batch_idx):
        rgb, depth, label = batch["rgb"], batch["depth"], batch["label"]
        label = sanitize_labels(label, num_classes=self.hparams.num_classes, ignore_index=255)
        student_logits = self(rgb, depth)
        self.teacher.eval()
        with torch.no_grad():
            teacher_logits = self.teacher(rgb, depth)
        ce_loss = self.train_criterion(student_logits, label)
        kd_loss, kd_stats = self.correct_entropy_kl_loss(
            student_logits,
            teacher_logits,
            label,
            self.kd_temperature,
            self.kd_entropy_threshold,
            ignore_index=255,
        )
        loss = ce_loss + self.kd_weight * kd_loss
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/ce_loss", ce_loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train/kd_loss", kd_loss, prog_bar=False, on_step=True, on_epoch=True)
        for name, value in kd_stats.items():
            self.log(f"train/{name}", value, prog_bar=False, on_step=True, on_epoch=True)
        return loss
