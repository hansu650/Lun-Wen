import torch
import torch.nn as nn
import torch.nn.functional as F


class CGPCLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.1,
        stage: str = "c3",
        min_pixels_per_class: int = 10,
        max_pixels_per_class: int = 128,
        detach_prototype: bool = True,
        ignore_index: int = 255,
        eps: float = 1e-6,
    ):
        super().__init__()
        if stage not in {"c2", "c3", "c4"}:
            raise ValueError("stage must be one of: c2, c3, c4")
        self.temperature = float(temperature)
        self.stage = stage
        self.min_pixels_per_class = int(min_pixels_per_class)
        self.max_pixels_per_class = int(max_pixels_per_class)
        self.detach_prototype = bool(detach_prototype)
        self.ignore_index = int(ignore_index)
        self.eps = float(eps)
        self.stage_to_index = {"c2": 1, "c3": 2, "c4": 3}
        self.last_stats = {}

    def forward(self, fused_feats, label):
        feat = fused_feats[self.stage_to_index[self.stage]]
        _, channels, height, width = feat.shape

        label_down = F.interpolate(
            label.unsqueeze(1).float(),
            size=(height, width),
            mode="nearest",
        ).squeeze(1).long()

        feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, channels)
        label_flat = label_down.reshape(-1)
        valid = label_flat != self.ignore_index
        feat_valid = feat_flat[valid]
        label_valid = label_flat[valid]

        prototypes = []
        queries = []
        query_targets = []
        prototype_labels = []

        for class_id in torch.unique(label_valid):
            class_mask = label_valid == class_id
            class_indices = torch.nonzero(class_mask, as_tuple=False).squeeze(1)
            if class_indices.numel() < self.min_pixels_per_class:
                continue
            if class_indices.numel() > self.max_pixels_per_class:
                perm = torch.randperm(class_indices.numel(), device=class_indices.device)
                class_indices = class_indices[perm[: self.max_pixels_per_class]]

            class_queries = feat_valid[class_indices]
            prototype = class_queries.mean(dim=0)
            if self.detach_prototype:
                prototype = prototype.detach()

            prototype_index = len(prototypes)
            prototypes.append(prototype)
            queries.append(class_queries)
            query_targets.append(
                torch.full(
                    (class_queries.shape[0],),
                    prototype_index,
                    device=feat.device,
                    dtype=torch.long,
                )
            )
            prototype_labels.append(class_id)

        if len(prototypes) < 2:
            zero = feat.sum() * 0.0
            self.last_stats = {
                "cgpc_loss": zero.detach(),
                "cgpc_num_classes": torch.zeros((), device=feat.device),
                "cgpc_num_queries": torch.zeros((), device=feat.device),
                "cgpc_stage": self.stage,
            }
            return zero

        prototype_tensor = F.normalize(torch.stack(prototypes, dim=0), dim=1, eps=self.eps)
        query_tensor = F.normalize(torch.cat(queries, dim=0), dim=1, eps=self.eps)
        target_tensor = torch.cat(query_targets, dim=0)

        logits = query_tensor @ prototype_tensor.t()
        logits = logits / self.temperature
        loss = F.cross_entropy(logits, target_tensor)

        self.last_stats = {
            "cgpc_loss": loss.detach(),
            "cgpc_num_classes": torch.tensor(float(len(prototype_labels)), device=feat.device),
            "cgpc_num_queries": torch.tensor(float(target_tensor.numel()), device=feat.device),
            "cgpc_stage": self.stage,
        }
        return loss
