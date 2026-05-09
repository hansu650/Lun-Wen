"""Cross-modal contrastive auxiliary losses."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalInfoNCELoss(nn.Module):
    def __init__(
        self,
        primary_channels,
        depth_channels,
        proj_dim=64,
        temperature=0.1,
        sample_points=256,
        stage_weights=(0.0, 0.0, 1.0, 1.0),
        eps=1e-8,
    ):
        super().__init__()
        self.proj_dim = int(proj_dim)
        self.temperature = float(temperature)
        self.sample_points = int(sample_points)
        self.eps = float(eps)
        self.primary_proj = nn.ModuleList([
            nn.Conv2d(ch, self.proj_dim, kernel_size=1, bias=False)
            for ch in primary_channels
        ])
        self.depth_proj = nn.ModuleList([
            nn.Conv2d(ch, self.proj_dim, kernel_size=1, bias=False)
            for ch in depth_channels
        ])
        weights = torch.tensor(stage_weights, dtype=torch.float32)
        if float(weights.sum().item()) <= 0:
            raise ValueError("stage_weights must contain at least one positive value")
        self.register_buffer("stage_weights", weights, persistent=False)

    def _stage_loss(self, primary_feat, depth_feat, primary_proj, depth_proj):
        assert primary_feat.shape[-2:] == depth_feat.shape[-2:]
        key = primary_proj(primary_feat.detach())
        query = depth_proj(depth_feat)
        key = F.normalize(key, dim=1, eps=self.eps)
        query = F.normalize(query, dim=1, eps=self.eps)

        B, _, H, W = query.shape
        sample_count = min(self.sample_points, H * W)
        idx = torch.randperm(H * W, device=query.device)[:sample_count]
        query_flat = query.flatten(2).transpose(1, 2)[:, idx, :].reshape(B * sample_count, self.proj_dim)
        key_flat = key.flatten(2).transpose(1, 2)[:, idx, :].reshape(B * sample_count, self.proj_dim)

        logits = query_flat @ key_flat.t() / self.temperature
        labels = torch.arange(B * sample_count, device=logits.device)
        return F.cross_entropy(logits, labels)

    def forward(self, primary_feats, depth_feats):
        total = primary_feats[0].new_zeros(())
        loss_dict = {}
        weights = self.stage_weights.to(device=primary_feats[0].device, dtype=primary_feats[0].dtype)

        for idx, (primary_feat, depth_feat, primary_proj, depth_proj, weight) in enumerate(
            zip(primary_feats, depth_feats, self.primary_proj, self.depth_proj, weights),
            start=1,
        ):
            if float(weight.item()) <= 0:
                loss_dict[f"contrast/stage{idx}"] = primary_feat.new_zeros(())
                continue
            stage_loss = self._stage_loss(primary_feat, depth_feat, primary_proj, depth_proj)
            total = total + weight * stage_loss
            loss_dict[f"contrast/stage{idx}"] = stage_loss.detach()

        total = total / weights.sum().clamp_min(self.eps)
        loss_dict["contrast/total"] = total.detach()
        return total, loss_dict
