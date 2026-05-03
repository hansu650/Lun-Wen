"""统一训练脚本"""
import os
import argparse
import warnings

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")
warnings.filterwarnings(
    "ignore",
    message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
    module=r"lightning\.pytorch\.utilities\._pytree",
)

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from src.data_module import NYUDataModule
from src.models.early_fusion import LitEarlyFusion
from src.models.mid_fusion import LitMidFusion, LitDFormerV2MidFusion


MODEL_REGISTRY = {
    "early": LitEarlyFusion,
    "mid_fusion": LitMidFusion,
    "dformerv2_mid_fusion": LitDFormerV2MidFusion,
}


def build_parser():
    parser = argparse.ArgumentParser(description="RGB-D Semantic Segmentation Training")
    parser.add_argument("--model", type=str, default="mid_fusion", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--data_root", type=str, required=True, help="NYU Depth V2 数据集根目录")
    parser.add_argument("--num_classes", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--early_stop_patience", type=int, default=15)
    parser.add_argument("--devices", type=str, default="1")
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--dformerv2_pretrained", type=str, default=None)
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
    model_cls = MODEL_REGISTRY[args.model]
    if args.model == "dformerv2_mid_fusion":
        return model_cls(
            num_classes=args.num_classes,
            lr=args.lr,
            dformerv2_pretrained=args.dformerv2_pretrained,
        )
    return model_cls(num_classes=args.num_classes, lr=args.lr)


def build_callbacks(args, monitor_metric: str):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename=f"{args.model}" + "-{epoch:02d}-{" + monitor_metric + ":.4f}",
        monitor=monitor_metric,
        mode="max",
        save_top_k=1,
    )
    early_stop_callback = EarlyStopping(
        monitor=monitor_metric,
        patience=args.early_stop_patience,
        mode="max",
    )
    return checkpoint_callback, early_stop_callback


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
    torch.set_float32_matmul_precision("high")
    monitor_metric = "val/mIoU"
    datamodule = build_datamodule(args)
    model = build_model(args)
    checkpoint_callback, early_stop_callback = build_callbacks(args, monitor_metric)
    trainer = build_trainer(args, callbacks=[checkpoint_callback, early_stop_callback])
    print(f"开始训练模型: {args.model}")
    trainer.fit(model, datamodule=datamodule)
    best_score = checkpoint_callback.best_model_score
    best_score_text = "N/A" if best_score is None else f"{best_score:.4f}"
    print(f"训练完成！最优模型: {checkpoint_callback.best_model_path}")
    print(f"最优 {monitor_metric}: {best_score_text}")


if __name__ == "__main__":
    main()
