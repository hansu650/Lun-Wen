"""统一训练入口。

这一版保持原来的工程骨架，只保留当前仓库里真实存在且能训练的模型：
- early
- mid_fusion
"""

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
from src.models.early_fusion import LitEarlyFusion
from src.models.mid_fusion import LitMidFusion

MODEL_REGISTRY = {
    "early": LitEarlyFusion,
    "mid_fusion": LitMidFusion,
}


def parse_bool_flag(value: str):
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


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

    # 这些参数先保留下来，避免已有脚本/命令因为少参数而报错。
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--min_lr_ratio", type=float, default=0.05)
    parser.add_argument("--backbone_lr_mult", type=float, default=0.1)
    parser.add_argument("--eval_tta", type=parse_bool_flag, default=False)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
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
    return model_cls(num_classes=args.num_classes, lr=args.lr)


def build_callbacks(args, monitor_metric: str):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        # 文件名里把 "/" 改成 "_"，避免 Windows 把它当成子目录。
        filename=f"{args.model}" + "-{epoch:02d}-{" + monitor_metric.replace("/", "_") + ":.4f}",
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
        check_val_every_n_epoch=args.check_val_every_n_epoch,
    )


def main():
    args = build_parser().parse_args()
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
