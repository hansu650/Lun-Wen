"""推理与可视化脚本。"""

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

import matplotlib.pyplot as plt
import torch

from src.data_module import AlbumentationsTransform, NYUDataset, get_val_transform
from src.models.early_fusion import LitEarlyFusion
from src.models.mid_fusion import LitMidFusion
from src.utils.visualize import visualize_prediction

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


def main():
    parser = argparse.ArgumentParser(description="RGB-D Semantic Segmentation Inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, default="mid_fusion", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--num_vis", type=int, default=5, help="可视化样本数")
    parser.add_argument("--save_dir", type=str, default="./visualizations")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--eval_tta", type=parse_eval_tta_flag, default="auto")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model_class = MODEL_REGISTRY[args.model]
    model = model_class.load_from_checkpoint(args.checkpoint)
    model.eval()
    model.to(device)

    val_transform = AlbumentationsTransform(get_val_transform())
    dataset = NYUDataset(args.data_root, split="test", transform=val_transform)

    os.makedirs(args.save_dir, exist_ok=True)

    for i in range(min(args.num_vis, len(dataset))):
        sample = dataset[i]
        rgb = sample["rgb"].unsqueeze(0).to(device)
        depth = sample["depth"].unsqueeze(0).to(device)
        gt = sample["label"]

        with torch.no_grad():
            logits = model(rgb, depth)
            if hasattr(model, "_eval_logits"):
                logits = model._eval_logits(logits, rgb, depth)
            pred = logits.argmax(dim=1).squeeze(0).cpu()

        fig = visualize_prediction(rgb.squeeze(0).cpu(), pred, gt)
        save_path = os.path.join(args.save_dir, f"pred_{i:03d}.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {save_path}")

    print(f"所有可视化结果已保存至: {args.save_dir}")


if __name__ == "__main__":
    main()
