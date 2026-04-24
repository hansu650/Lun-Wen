import torch
import torch.nn as nn
import torch.distributed as dist
import lightning as L

from ..utils.metrics import compute_miou, sanitize_labels


class BaseLitSeg(L.LightningModule):
    def __init__(self, num_classes=40, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.register_buffer("_val_confmat", torch.zeros(num_classes, num_classes, dtype=torch.long), persistent=False)

    def forward(self, rgb, depth):
        return self.model(rgb, depth)

    def _eval_logits(self, logits, rgb, depth):
        return logits

    def training_step(self, batch, batch_idx):
        rgb, depth, label = batch["rgb"], batch["depth"], batch["label"]
        label = sanitize_labels(label, num_classes=self.hparams.num_classes, ignore_index=255)
        logits = self(rgb, depth)
        loss = self.criterion(logits, label)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        rgb, depth, label = batch["rgb"], batch["depth"], batch["label"]
        label = sanitize_labels(label, num_classes=self.hparams.num_classes, ignore_index=255)
        logits = self(rgb, depth)
        logits_eval = self._eval_logits(logits, rgb, depth)
        loss = self.criterion(logits, label)
        pred = logits_eval.argmax(dim=1)
        miou_batch = compute_miou(pred, label, num_classes=self.hparams.num_classes)
        valid = label != 255
        pred_valid = pred[valid]
        gt = label[valid]
        if gt.numel() > 0:
            n = self.hparams.num_classes
            hist = torch.bincount(gt * n + pred_valid, minlength=n * n).reshape(n, n)
            self._val_confmat += hist.to(self._val_confmat.device).long()
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/mIoU_batch", miou_batch, prog_bar=False, on_step=False, on_epoch=True)
        return {"val_loss": loss, "val_miou_batch": miou_batch}

    def on_validation_epoch_start(self):
        self._val_confmat.zero_()

    def on_validation_epoch_end(self):
        conf = self._val_confmat.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(conf, op=dist.ReduceOp.SUM)
        inter = torch.diag(conf).float()
        union = conf.sum(dim=1).float() + conf.sum(dim=0).float() - inter
        miou_global = (inter / union.clamp_min(1.0)).mean()
        self.log("val/mIoU_global", miou_global, prog_bar=False, on_step=False, on_epoch=True, sync_dist=False)
        self.log("val/mIoU", miou_global, prog_bar=True, on_step=False, on_epoch=True, sync_dist=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
