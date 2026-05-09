import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, ignore_index: int = 255, smooth: float = 1.0, eps: float = 1e-7):
        super().__init__()
        self.ignore_index = int(ignore_index)
        self.smooth = float(smooth)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        target = target.long()
        valid = target != self.ignore_index
        if valid.sum() == 0:
            return logits.sum() * 0.0

        target_fixed = target.clone()
        target_fixed[~valid] = 0

        probs = torch.softmax(logits, dim=1)
        target_one_hot = F.one_hot(target_fixed, num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).to(dtype=logits.dtype)
        valid_mask = valid.unsqueeze(1).to(dtype=logits.dtype)

        probs = probs * valid_mask
        target_one_hot = target_one_hot * valid_mask

        dims = (0, 2, 3)
        intersection = (probs * target_one_hot).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + target_one_hot.sum(dim=dims)
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth + self.eps)
        return 1.0 - dice_score.mean()


class CEDiceLoss(nn.Module):
    def __init__(self, ignore_index: int = 255, dice_weight: float = 0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)
        self.dice_weight = float(dice_weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, target)
        dice_loss = self.dice(logits, target)
        return ce_loss + self.dice_weight * dice_loss
