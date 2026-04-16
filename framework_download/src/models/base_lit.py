import torch
import torch.nn as nn
import torch.distributed as dist
import lightning as L

from ..utils.metrics import compute_miou, sanitize_labels
# 整理标签和算miou的

# 分割训练通用骨架 base lightning segmentation
class BaseLitSeg(L.LightningModule):
    def __init__(self, num_classes=40, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()# 参数保存一下
        self.criterion = nn.CrossEntropyLoss(ignore_index=255) # 交叉熵，如果预测的好就小,看你有多自信
        self.register_buffer("_val_confmat", torch.zeros(num_classes, num_classes, dtype=torch.long), persistent=False)
# 算miou的
    def forward(self, rgb, depth):
        return self.model(rgb, depth)

    def _eval_logits(self, logits, rgb, depth):# 对logits最额外处理
        return logits

    def training_step(self, batch, batch_idx):
        rgb, depth, label = batch["rgb"], batch["depth"], batch["label"]
        label = sanitize_labels(label, num_classes=self.hparams.num_classes, ignore_index=255)# 无效标签变成255
        logits = self(rgb, depth)
        loss = self.criterion(logits, label)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss# 打开日志

    def validation_step(self, batch, batch_idx):
        rgb, depth, label = batch["rgb"], batch["depth"], batch["label"]
        label = sanitize_labels(label, num_classes=self.hparams.num_classes, ignore_index=255)
        logits = self(rgb, depth)
        logits_eval = self._eval_logits(logits, rgb, depth)
        loss = self.criterion(logits, label)
        pred = logits_eval.argmax(dim=1) # 取最大值得到类别
        miou_batch = compute_miou(pred, label, num_classes=self.hparams.num_classes)
        valid = label != 255
        pred_valid = pred[valid]
        gt = label[valid]
        if gt.numel() > 0: # 有有效像素值就继续统计
            n = self.hparams.num_classes
            hist = torch.bincount(gt * n + pred_valid, minlength=n * n).reshape(n, n)
            self._val_confmat += hist.to(self._val_confmat.device).long()
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/mIoU_batch", miou_batch, prog_bar=False, on_step=False, on_epoch=True)
        return {"val_loss": loss, "val_miou_batch": miou_batch}
# hist 混淆矩阵，列预测行真实，所以行==列才是真正的正确结果，其他都是预测错误的结果
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

# miou 是通过 IoU_k = intersection_k / union_k
# mIoU = mean(IoU_1, IoU_2, ..., IoU_n)
# union和intersection都是通过混淆矩阵算出来的