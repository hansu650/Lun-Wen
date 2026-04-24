"""评估指标工具"""
import torch


def sanitize_labels(target, num_classes=40, ignore_index=255):
    target = target.long()
    if target.numel() > 0:
        tmin = int(target.min().item())
        tmax = int(target.max().item())
        if tmin >= 1 and tmax <= num_classes:
            target = target - 1
    invalid = (target < 0) | (target >= num_classes)
    if invalid.any():
        target = target.masked_fill(invalid, ignore_index)
    return target


def compute_miou(pred, target, num_classes=40, ignore_index=255):
    """计算 mean IoU"""
    target = sanitize_labels(target, num_classes=num_classes, ignore_index=ignore_index)
    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]
    
    if target.numel() == 0:
        return 0.0
    
    hist = torch.bincount(
        target * num_classes + pred,
        minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes)
    
    inter = torch.diag(hist).float()
    union = hist.sum(1).float() + hist.sum(0).float() - inter
    iou = inter / union.clamp_min(1.0)
    
    return iou.mean().item()


def compute_pixel_accuracy(pred, target, ignore_index=255):
    """计算像素准确率"""
    target = sanitize_labels(target, num_classes=40, ignore_index=ignore_index)
    mask = target != ignore_index
    correct = (pred[mask] == target[mask]).float().sum()
    total = mask.sum()
    return (correct / total.clamp_min(1.0)).item()
