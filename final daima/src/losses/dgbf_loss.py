import torch
import torch.nn as nn
import torch.nn.functional as F


class DGBFLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        mode: str = "depth_semantic",
        ignore_index: int = 255,
        eps: float = 1e-6,
    ):
        super().__init__()
        if mode not in {"depth_semantic", "semantic_only", "depth_only", "focal_only", "none"}:
            raise ValueError("mode must be one of: depth_semantic, semantic_only, depth_only, focal_only, none")
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.mode = mode
        self.ignore_index = int(ignore_index)
        self.eps = float(eps)
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
        self.last_stats = {}

    def _depth_edge(self, depth, size):
        if depth.shape[-2:] != size:
            depth = F.interpolate(depth, size=size, mode="bilinear", align_corners=False)
        depth = depth.float()
        grad_x = F.conv2d(depth, self.sobel_x.to(dtype=depth.dtype), padding=1)
        grad_y = F.conv2d(depth, self.sobel_y.to(dtype=depth.dtype), padding=1)
        edge = torch.sqrt(grad_x.square() + grad_y.square() + self.eps).squeeze(1)
        flat = edge.flatten(1)
        edge_min = flat.min(dim=1).values.view(-1, 1, 1)
        edge_max = flat.max(dim=1).values.view(-1, 1, 1)
        return (edge - edge_min) / (edge_max - edge_min + self.eps)

    def _semantic_edge(self, target):
        valid = target != self.ignore_index
        semantic = torch.zeros_like(target, dtype=torch.bool)

        diff_h = (target[:, :, 1:] != target[:, :, :-1]) & valid[:, :, 1:] & valid[:, :, :-1]
        semantic[:, :, 1:] |= diff_h
        semantic[:, :, :-1] |= diff_h

        diff_v = (target[:, 1:, :] != target[:, :-1, :]) & valid[:, 1:, :] & valid[:, :-1, :]
        semantic[:, 1:, :] |= diff_v
        semantic[:, :-1, :] |= diff_v

        return semantic.float()

    def forward(self, logits, target, depth):
        ce = F.cross_entropy(
            logits,
            target,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        valid = target != self.ignore_index
        valid_count = valid.sum().clamp_min(1)

        if self.mode == "none":
            weight = torch.ones_like(ce)
            boundary = torch.zeros_like(ce)
        else:
            depth_edge = self._depth_edge(depth, target.shape[-2:])
            semantic_edge = self._semantic_edge(target)

            if self.mode == "depth_semantic":
                boundary = depth_edge * semantic_edge
            elif self.mode == "semantic_only":
                boundary = semantic_edge
            elif self.mode == "depth_only":
                boundary = depth_edge
            elif self.mode == "focal_only":
                boundary = torch.ones_like(ce)

            safe_target = target.masked_fill(~valid, 0)
            probs = F.softmax(logits.detach(), dim=1)
            p_t = probs.gather(1, safe_target.unsqueeze(1)).squeeze(1)
            focal_mod = (1.0 - p_t).pow(self.gamma)
            weight = 1.0 + self.alpha * boundary * focal_mod

        valid_f = valid.float()
        loss = (weight * ce * valid_f).sum() / valid_count
        with torch.no_grad():
            masked_boundary = boundary[valid]
            masked_weight = weight[valid]
            self.last_stats = {
                "dgbf_boundary_mean": masked_boundary.mean() if masked_boundary.numel() > 0 else loss.detach() * 0,
                "dgbf_boundary_max": masked_boundary.max() if masked_boundary.numel() > 0 else loss.detach() * 0,
                "dgbf_weight_mean": masked_weight.mean() if masked_weight.numel() > 0 else loss.detach() * 0,
                "dgbf_weight_max": masked_weight.max() if masked_weight.numel() > 0 else loss.detach() * 0,
            }
        return loss
