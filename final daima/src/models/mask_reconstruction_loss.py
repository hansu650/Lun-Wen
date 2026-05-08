"""Training-only feature-level cross-modal mask reconstruction loss."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureMaskReconstructionLoss(nn.Module):
    def __init__(
        self,
        primary_channels,
        depth_channels,
        stage_weights=None,
        mask_ratio_depth=0.30,
        mask_ratio_primary=0.15,
        alpha=0.5,
        loss_type="smooth_l1",
    ):
        super().__init__()
        if stage_weights is None:
            raise ValueError("stage_weights must be explicitly provided with 4 values")
        if len(primary_channels) != 4 or len(depth_channels) != 4 or len(stage_weights) != 4:
            raise ValueError("primary_channels, depth_channels, and stage_weights must have length 4")

        self.mask_ratio_depth = float(mask_ratio_depth)
        self.mask_ratio_primary = float(mask_ratio_primary)
        self.alpha = float(alpha)
        self.loss_fn = {
            "smooth_l1": F.smooth_l1_loss,
            "l1": F.l1_loss,
            "mse": F.mse_loss,
        }[loss_type]
        self.eps = 1e-6
        self.register_buffer(
            "stage_weights",
            torch.tensor(stage_weights, dtype=torch.float32),
            persistent=False,
        )

        self.primary_to_depth_heads = nn.ModuleList()
        self.depth_to_primary_heads = nn.ModuleList()
        for primary_ch, depth_ch in zip(primary_channels, depth_channels):
            in_ch = primary_ch + depth_ch
            hidden_ch = min(256, max(in_ch // 2, 64))
            self.primary_to_depth_heads.append(self._make_head(in_ch, hidden_ch, depth_ch))
            self.depth_to_primary_heads.append(self._make_head(in_ch, hidden_ch, primary_ch))

    @staticmethod
    def _make_head(in_channels, hidden_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def _make_mask(self, feat, ratio):
        b, _, h, w = feat.shape
        return (torch.rand((b, 1, h, w), device=feat.device, dtype=feat.dtype) < ratio).to(dtype=feat.dtype)

    def _masked_loss(self, pred, target, mask):
        per_pixel_loss = self.loss_fn(pred, target.detach(), reduction="none")
        denominator = mask.sum().clamp_min(self.eps) * pred.shape[1]
        return (per_pixel_loss * mask).sum() / denominator

    def forward(self, primary_feats, depth_feats):
        depth_total = primary_feats[0].new_zeros(())
        primary_total = primary_feats[0].new_zeros(())
        depth_mask_total = primary_feats[0].new_zeros(())
        primary_mask_total = primary_feats[0].new_zeros(())
        loss_dict = {}
        weights = self.stage_weights.to(device=primary_feats[0].device, dtype=primary_feats[0].dtype)

        for stage_idx, weight in enumerate(weights):
            stage_name = f"maskrec/stage{stage_idx + 1}"
            if float(weight.item()) <= 0:
                loss_dict[stage_name] = primary_feats[stage_idx].new_zeros(())
                continue

            primary_feat = primary_feats[stage_idx]
            depth_feat = depth_feats[stage_idx]

            depth_mask = self._make_mask(depth_feat, self.mask_ratio_depth)
            masked_depth = depth_feat * (1 - depth_mask)
            pred_depth = self.primary_to_depth_heads[stage_idx](torch.cat([primary_feat, masked_depth], dim=1))
            depth_loss = self._masked_loss(pred_depth, depth_feat, depth_mask)

            primary_mask = self._make_mask(primary_feat, self.mask_ratio_primary)
            masked_primary = primary_feat * (1 - primary_mask)
            pred_primary = self.depth_to_primary_heads[stage_idx](torch.cat([depth_feat, masked_primary], dim=1))
            primary_loss = self._masked_loss(pred_primary, primary_feat, primary_mask)

            stage_loss = depth_loss + self.alpha * primary_loss
            loss_dict[stage_name] = stage_loss.detach()

            depth_total = depth_total + weight * depth_loss
            primary_total = primary_total + weight * primary_loss
            depth_mask_total = depth_mask_total + weight * depth_mask.mean()
            primary_mask_total = primary_mask_total + weight * primary_mask.mean()

        weight_sum = weights.sum().clamp_min(self.eps)
        depth_total = depth_total / weight_sum
        primary_total = primary_total / weight_sum
        total = depth_total + self.alpha * primary_total

        loss_dict["maskrec/depth_rec"] = depth_total.detach()
        loss_dict["maskrec/primary_rec"] = primary_total.detach()
        loss_dict["maskrec/total"] = total.detach()
        loss_dict["maskrec/depth_mask_ratio"] = (depth_mask_total / weight_sum).detach()
        loss_dict["maskrec/primary_mask_ratio"] = (primary_mask_total / weight_sum).detach()
        return total, loss_dict
