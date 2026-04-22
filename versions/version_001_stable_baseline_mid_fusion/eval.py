"""评估脚本。"""

import argparse
import os
import warnings

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")
warnings.filterwarnings(
    "ignore",
    message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
    module=r"lightning\.pytorch\.utilities\._pytree",
)

import torch
from torch.utils.data import DataLoader

from src.data_module import AlbumentationsTransform, NYUDataset, get_val_transform
from src.models.early_fusion import LitEarlyFusion
from src.models.mid_fusion import LitMidFusion
from src.utils.metrics import sanitize_labels

MODEL_REGISTRY = {
    "early": LitEarlyFusion,
    "mid_fusion": LitMidFusion,
}


def parse_eval_tta_flag(value: str):
    value = str(value).strip().lower()
    if value in {"auto", "default"}:
        return "auto"
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid eval_tta value: {value}")


def evaluate(model, dataloader, device="cuda"):
    model.eval()
    model.to(device)

    num_classes = int(model.hparams.num_classes)
    confmat = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)
    correct = torch.tensor(0.0, device=device)
    total = torch.tensor(0.0, device=device)

    with torch.no_grad():
        for batch in dataloader:
            rgb = batch["rgb"].to(device)
            depth = batch["depth"].to(device)
            label = sanitize_labels(
                batch["label"].to(device),
                num_classes=num_classes,
                ignore_index=255,
            )

            logits = model(rgb, depth)
            if hasattr(model, "_eval_logits"):
                logits = model._eval_logits(logits, rgb, depth)
            pred = logits.argmax(dim=1)

            valid = label != 255
            if valid.any():
                pred_valid = pred[valid]
                gt_valid = label[valid]
                hist = torch.bincount(
                    gt_valid * num_classes + pred_valid,
                    minlength=num_classes * num_classes,
                ).reshape(num_classes, num_classes)
                confmat += hist
                correct += (pred_valid == gt_valid).float().sum()
                total += gt_valid.numel()

    inter = torch.diag(confmat).float()
    union = confmat.sum(dim=1).float() + confmat.sum(dim=0).float() - inter
    miou = (inter / union.clamp_min(1.0)).mean().item()
    pix_acc = (correct / total.clamp_min(1.0)).item()
    return {
        "mIoU": miou,
        "PixelAcc": pix_acc,
    }


def main():
    parser = argparse.ArgumentParser(description="RGB-D Semantic Segmentation Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--model", type=str, default="mid_fusion", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_tta", type=parse_eval_tta_flag, default="auto")
    args = parser.parse_args()

    model_class = MODEL_REGISTRY[args.model]
    model = model_class.load_from_checkpoint(args.checkpoint)

    val_transform = AlbumentationsTransform(get_val_transform())
    val_dataset = NYUDataset(args.data_root, split="test", transform=val_transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = evaluate(model, val_loader, device=device)
    print(f"评估结果: mIoU={metrics['mIoU']:.4f}, PixelAcc={metrics['PixelAcc']:.4f}")


if __name__ == "__main__":
    main()
