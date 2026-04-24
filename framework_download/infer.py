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
from src.models.mid_fusion import LitMidFusion
from src.utils.visualize import visualize_prediction


MODEL_REGISTRY = {"mid_fusion": LitMidFusion}


def main():
    parser = argparse.ArgumentParser(description="RGB-D Semantic Segmentation Inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, default="mid_fusion", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--num_vis", type=int, default=5)
    parser.add_argument("--save_dir", type=str, default="./visualizations")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = MODEL_REGISTRY[args.model].load_from_checkpoint(args.checkpoint)
    model.eval()
    model.to(device)

    dataset = NYUDataset(args.data_root, split="test", transform=AlbumentationsTransform(get_val_transform()))
    os.makedirs(args.save_dir, exist_ok=True)

    for index in range(min(args.num_vis, len(dataset))):
        sample = dataset[index]
        rgb = sample["rgb"].unsqueeze(0).to(device)
        depth = sample["depth"].unsqueeze(0).to(device)
        gt = sample["label"]

        with torch.no_grad():
            logits = model(rgb, depth)
            logits = model._eval_logits(logits, rgb, depth)
            pred = logits.argmax(dim=1).squeeze(0).cpu()

        fig = visualize_prediction(rgb.squeeze(0).cpu(), pred, gt)
        save_path = os.path.join(args.save_dir, f"pred_{index:03d}.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
