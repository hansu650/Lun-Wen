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

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from src.data_module import NYUDataModule
from src.models.mid_fusion import LitMidFusion


MODEL_REGISTRY = {"mid_fusion": LitMidFusion}


def build_parser():
    parser = argparse.ArgumentParser(description="RGB-D Semantic Segmentation Training")
    parser.add_argument("--model", type=str, default="mid_fusion", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--early_stop_patience", type=int, default=15)
    parser.add_argument("--devices", type=str, default="1")
    parser.add_argument("--accelerator", type=str, default="auto")
    return parser


def parse_devices(devices: str):
    if devices == "auto":
        return "auto"
    try:
        return int(devices)
    except ValueError:
        return devices


def build_datamodule(args):
    return NYUDataModule(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


def build_model(args):
    return MODEL_REGISTRY[args.model](num_classes=args.num_classes, lr=args.lr)


def build_callbacks(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename=args.model + "-{epoch:02d}-{val_mIoU:.4f}",
        monitor="val_mIoU",
        mode="max",
        save_top_k=1,
    )
    early_stop = EarlyStopping(
        monitor="val_mIoU",
        patience=args.early_stop_patience,
        mode="max",
    )
    return checkpoint, early_stop


def build_trainer(args, callbacks):
    return L.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=parse_devices(args.devices),
        callbacks=list(callbacks),
        default_root_dir=args.checkpoint_dir,
        log_every_n_steps=10,
    )


def main():
    args = build_parser().parse_args()
    datamodule = build_datamodule(args)
    model = build_model(args)
    checkpoint, early_stop = build_callbacks(args)
    trainer = build_trainer(args, callbacks=[checkpoint, early_stop])

    print(f"Start training: {args.model}")
    trainer.fit(model, datamodule=datamodule)

    best_score = checkpoint.best_model_score
    best_score_text = "N/A" if best_score is None else f"{best_score:.4f}"
    print(f"Best checkpoint: {checkpoint.best_model_path}")
    print(f"Best val/mIoU: {best_score_text}")


if __name__ == "__main__":
    main()
