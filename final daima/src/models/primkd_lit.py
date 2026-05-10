"""Logit-only PMAD/PrimKD distillation Lightning module."""
import torch
import torch.nn.functional as F

from .base_lit import BaseLitSeg
from .mid_fusion import DFormerV2MidFusionSegmentor
from .teacher_model import DFormerV2RGBTeacherSegmentor
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
        self.teacher = DFormerV2RGBTeacherSegmentor(num_classes=num_classes)
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
            teacher_logits = self.teacher(rgb)
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
