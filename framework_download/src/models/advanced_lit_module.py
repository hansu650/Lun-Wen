"""Advanced RGB-D model adapted to the local training framework."""
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.base_lit import BaseLitSeg


class MockViTBackbone(nn.Module):
    """A compact ViT-style backbone for teaching/demo purposes."""

    def __init__(self, embed_dim=384, patch_size=14):
        super().__init__()
        self.patch_size = patch_size  # 每个 patch 的边长
        self.embed_dim = embed_dim  # 每个 patch token 的特征维度

        # Patch Embedding：
        # 用一个 stride=patch_size 的卷积，把整张图切成 patch 并投影成 token 特征。
        self.patch_embed = nn.Conv2d(
            3,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # 这里用几层 TransformerEncoderLayer 模拟 ViT 主体，
        # 是课程 06 的教学版，不是真正的大规模预训练 ViT。
        self.blocks = nn.Sequential(
            *[
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=6,
                    dim_feedforward=embed_dim * 4,
                    batch_first=True,
                )
                for _ in range(4)
            ]
        )

    def forward(self, x):
        batch, _, height, width = x.shape
        # 1. 切 patch 并做线性投影
        x = self.patch_embed(x)
        # 2. 从 [B, C, H/P, W/P] 变成 [B, num_patches, C]
        x = x.flatten(2).transpose(1, 2)
        # 3. 用 Transformer 在 patch token 序列上建模全局关系
        x = self.blocks(x)
        # 4. 再变回空间特征图，后面才能和 depth 分支融合
        x = x.transpose(1, 2).reshape(
            batch,
            self.embed_dim,
            height // self.patch_size,
            width // self.patch_size,
        )
        return [x]


class AdvancedRGBDSegmentor(nn.Module):
    """Teaching-friendly ViT RGB + CNN depth segmentor."""

    def __init__(self, num_classes=40, embed_dim=384):
        super().__init__()
        # RGB 这边走 ViT 风格 backbone，表示课程 06 里的高级路线
        self.rgb_backbone = MockViTBackbone(embed_dim=embed_dim)

        # Depth 这里还是保留一个较轻的 CNN 编码器。
        # 教学上这样更容易看清：变化主要发生在 RGB backbone 和训练策略上。
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        # 融合部分这里只做了一个简化版的拼接 + 1x1 Conv，
        # 课程里提到的更复杂 token 级 MFM 这里只是方向，还没在本地骨架里完全展开。
        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(self, rgb, depth):
        # RGB 走 ViT 风格特征，Depth 走 CNN 特征
        rgb_feat = self.rgb_backbone(rgb)[0]
        depth_feat = self.depth_encoder(depth)

        # 两个分支空间分辨率不同时，先对齐再融合
        if rgb_feat.shape[-2:] != depth_feat.shape[-2:]:
            depth_feat = F.interpolate(
                depth_feat,
                size=rgb_feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # 这里的 logits 还在低分辨率空间，最后要恢复回原图大小
        fused = self.fusion(torch.cat([rgb_feat, depth_feat], dim=1))
        logits = self.decoder(fused)
        return F.interpolate(
            logits,
            size=rgb.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )


class LitAdvancedRGBD(BaseLitSeg):
    def __init__(
        self,
        num_classes=40,
        lr=1e-4,
        weight_decay=1e-2,
        warmup_steps=1000,
        min_lr_ratio=0.05,
        backbone_lr_mult=0.1,
        eval_tta=False,
    ):
        super().__init__(num_classes=num_classes, lr=lr)
        self.save_hyperparameters()
        # 训练/验证流程仍然复用 BaseLitSeg，
        # 只是高级模型会额外改优化器和验证时的推理逻辑。
        self.model = AdvancedRGBDSegmentor(num_classes=num_classes)

    def _eval_logits(self, logits, rgb, depth):
        # 验证阶段如果开了 TTA，就不用原始 logits，
        # 而是改用多尺度 + 翻转之后平均得到的结果。
        if self.hparams.eval_tta:
            return self._predict_with_tta(rgb, depth)
        return logits

    @torch.no_grad()
    def _predict_with_tta(self, rgb, depth):
        height, width = rgb.shape[-2:]
        prob_sum = None
        num_preds = 0

        # 课程 06 里的 TTA 主线：
        # 多尺度推理 + 水平翻转 + 概率平均。
        for scale in (0.75, 1.0, 1.25, 1.5):
            scaled_h = max(32, int(round(height * scale)))
            scaled_w = max(32, int(round(width * scale)))

            rgb_scaled = F.interpolate(
                rgb,
                size=(scaled_h, scaled_w),
                mode="bilinear",
                align_corners=False,
            )
            depth_scaled = F.interpolate(
                depth,
                size=(scaled_h, scaled_w),
                mode="bilinear",
                align_corners=False,
            )

            # 先跑原图方向
            logits_scaled = self(rgb_scaled, depth_scaled)
            logits_scaled = F.interpolate(
                logits_scaled,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            probs = torch.softmax(logits_scaled, dim=1)
            prob_sum = probs if prob_sum is None else prob_sum + probs
            num_preds += 1

            # 再跑水平翻转版本，最后再翻回来做平均
            rgb_flip = torch.flip(rgb_scaled, dims=[3])
            depth_flip = torch.flip(depth_scaled, dims=[3])
            logits_flip = self(rgb_flip, depth_flip)
            logits_flip = torch.flip(logits_flip, dims=[3])
            logits_flip = F.interpolate(
                logits_flip,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            probs_flip = torch.softmax(logits_flip, dim=1)
            prob_sum += probs_flip
            num_preds += 1

        # 返回 log(prob)，这样后面既能 argmax，也能保持和“logits 风格”接口兼容。
        prob_mean = prob_sum / float(max(num_preds, 1))
        return torch.log(prob_mean.clamp_min(1e-8))

    def configure_optimizers(self):
        backbone_params = []
        head_params = []

        # 分层学习率：
        # RGB backbone 用较小 lr，其他任务相关模块用基础 lr。
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("model.rgb_backbone"):
                backbone_params.append(param)
            else:
                head_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": head_params,
                    "lr": self.hparams.lr,
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": backbone_params,
                    "lr": self.hparams.lr * self.hparams.backbone_lr_mult,
                    "weight_decay": self.hparams.weight_decay,
                },
            ]
        )

        # 这里直接按 step 级别做 Warmup + Cosine。
        # 因为 Lightning 已经能给出 estimated_stepping_batches，
        # 所以我们可以直接在当前训练总步数上构造调度器。
        total_steps = max(1, int(self.trainer.estimated_stepping_batches))
        warmup_steps = max(0, min(self.hparams.warmup_steps, total_steps - 1))
        min_lr_ratio = self.hparams.min_lr_ratio

        def lr_lambda(step):
            # 前期 warmup：慢慢升到目标 lr，避免一开始训练不稳
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)

            # 后期 cosine decay：慢慢把学习率降下来，帮助收敛
            progress = float(step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            cosine = 0.5 * (1.0 + math.cos(progress * math.pi))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
