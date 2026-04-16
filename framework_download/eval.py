"""评估脚本"""
import os
import argparse
import warnings
# 加载一个训练好的 checkpoint，在 test 集上算 mIoU 和 PixelAcc。
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")
warnings.filterwarnings(
    "ignore",
    message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
    module=r"lightning\.pytorch\.utilities\._pytree",
)# 关掉一些warnings

import torch
from torch.utils.data import DataLoader
# 模型相关类和计算miou啥的函数
from src.data_module import NYUDataModule, NYUDataset, AlbumentationsTransform, get_val_transform
from src.models.advanced_lit_module import LitAdvancedRGBD
from src.models.attention_fusion_model import LitAttentionFusion
from src.models.dformer_model import LitDFormerInspired
from src.models.early_fusion import LitEarlyFusion
from src.models.mid_fusion import LitMidFusion
from src.utils.metrics import compute_miou, compute_pixel_accuracy


MODEL_REGISTRY = {
    "early": LitEarlyFusion,
    "mid_fusion": LitMidFusion,
    "attention": LitAttentionFusion,
    "advanced": LitAdvancedRGBD,
    "dformer": LitDFormerInspired,
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

    total_miou = 0.0
    total_pix_acc = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            rgb = batch["rgb"].to(device)
            depth = batch["depth"].to(device)
            label = batch["label"].to(device)

            logits = model(rgb, depth)
            if hasattr(model, "_eval_logits"):
                logits = model._eval_logits(logits, rgb, depth)
            pred = logits.argmax(dim=1)

            total_miou += compute_miou(pred, label, num_classes=40)
            total_pix_acc += compute_pixel_accuracy(pred, label)
            num_batches += 1

    return {
        "mIoU": total_miou / num_batches,
        "PixelAcc": total_pix_acc / num_batches,
    }


def main():
    parser = argparse.ArgumentParser(description="RGB-D Semantic Segmentation Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--model", type=str, default="mid_fusion", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_tta", type=parse_eval_tta_flag, default="auto")
    args = parser.parse_args()

    ModelClass = MODEL_REGISTRY[args.model]
    model = ModelClass.load_from_checkpoint(args.checkpoint)
    if args.eval_tta != "auto" and hasattr(model, "hparams") and hasattr(model.hparams, "eval_tta"):
        model.hparams.eval_tta = args.eval_tta

    val_transform = AlbumentationsTransform(get_val_transform())
    val_dataset = NYUDataset(args.data_root, split="test", transform=val_transform) # 评估数据集
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = evaluate(model, val_loader, device=device)
    
    print(f"评估结果: mIoU={metrics['mIoU']:.4f}, PixelAcc={metrics['PixelAcc']:.4f}")


if __name__ == "__main__":
    main()
