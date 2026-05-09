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
warnings.filterwarnings(
    "ignore",
    message=r"Importing from timm\.models\.layers is deprecated.*",
    category=FutureWarning,
    module=r"timm\.models\.layers",
)
warnings.filterwarnings(
    "ignore",
    message=r"torch\.meshgrid: in an upcoming release.*",
    category=UserWarning,
    module=r"torch\.functional",
)

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping

from src.data_module import NYUDataModule
from src.models.early_fusion import LitEarlyFusion
from src.models.mid_fusion import (
    LitMidFusion,
    LitDFormerV2MidFusion,
    LitDFormerV2DepthFFTSelect,
    LitDFormerV2FFTFreqEnhance,
    LitDFormerV2MSFreqCov,
    LitDFormerV2FeatMaskRecC34,
    LitDFormerV2CMInfoNCE,
)


MODEL_REGISTRY = {
    "early": LitEarlyFusion,
    "mid_fusion": LitMidFusion,
    "dformerv2_mid_fusion": LitDFormerV2MidFusion,
    "dformerv2_depth_fft_select": LitDFormerV2DepthFFTSelect,
    "dformerv2_fft_freq_enhance": LitDFormerV2FFTFreqEnhance,
    "dformerv2_ms_freqcov": LitDFormerV2MSFreqCov,
    "dformerv2_feat_maskrec_c34": LitDFormerV2FeatMaskRecC34,
    "dformerv2_cm_infonce": LitDFormerV2CMInfoNCE,
}


class DirectStateDictCheckpoint(Callback):
    def __init__(self, dirpath, filename_prefix, monitor, mode="max"):
        super().__init__()
        self.dirpath = dirpath
        self.filename_prefix = filename_prefix
        self.monitor = monitor
        self.mode = mode
        self.best_model_score = None
        self.best_model_path = ""

    def _is_better(self, current):
        if self.best_model_score is None:
            return True
        if self.mode == "max":
            return current > self.best_model_score
        return current < self.best_model_score

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if self.monitor not in metrics:
            return
        current = float(metrics[self.monitor].detach().cpu())
        if not self._is_better(current):
            return
        self.best_model_score = current
        epoch = trainer.current_epoch
        monitor_name = self.monitor.replace("/", "_")
        filename = f"{self.filename_prefix}-epoch={epoch:02d}-{monitor_name}={current:.4f}.pt"
        filepath = os.path.join(self.dirpath, filename)
        torch.save(pl_module.state_dict(), filepath)
        if self.best_model_path and self.best_model_path != filepath and os.path.exists(self.best_model_path):
            os.remove(self.best_model_path)
        self.best_model_path = filepath


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
    parser.add_argument("--lambda_freq", type=float, default=0.01)
    parser.add_argument("--freq_eta", type=float, default=1.0)
    parser.add_argument("--freq_proj_dim", type=int, default=64)
    parser.add_argument("--freq_kernel_size", type=int, default=3)
    parser.add_argument("--freq_stage_weights", type=str, default="1,1,1,1")
    parser.add_argument("--cutoff_ratio", type=float, default=0.25)
    parser.add_argument("--gamma_init", type=float, default=0.05)
    parser.add_argument("--lambda_mask", type=float, default=0.01)
    parser.add_argument("--mask_ratio_depth", type=float, default=0.30)
    parser.add_argument("--mask_ratio_primary", type=float, default=0.15)
    parser.add_argument("--maskrec_alpha", type=float, default=0.5)
    parser.add_argument("--maskrec_loss_type", type=str, default="smooth_l1")
    parser.add_argument("--maskrec_stage_weights", type=str, default="1,1,1,1")
    parser.add_argument("--lambda_contrast", type=float, default=0.005)
    parser.add_argument("--contrast_temperature", type=float, default=0.1)
    parser.add_argument("--contrast_proj_dim", type=int, default=64)
    parser.add_argument("--contrast_sample_points", type=int, default=256)
    parser.add_argument("--contrast_stage_weights", type=str, default="0,0,1,1")
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
    if args.model == "dformerv2_ms_freqcov":
        freq_stage_weights = tuple(float(v.strip()) for v in args.freq_stage_weights.split(","))
        return model_cls(
            num_classes=args.num_classes,
            lr=args.lr,
            dformerv2_pretrained=args.dformerv2_pretrained,
            lambda_freq=args.lambda_freq,
            freq_eta=args.freq_eta,
            freq_proj_dim=args.freq_proj_dim,
            freq_kernel_size=args.freq_kernel_size,
            freq_stage_weights=freq_stage_weights,
        )
    if args.model == "dformerv2_feat_maskrec_c34":
        maskrec_stage_weights = tuple(float(v.strip()) for v in args.maskrec_stage_weights.split(","))
        if len(maskrec_stage_weights) != 4:
            raise ValueError("--maskrec_stage_weights must contain 4 comma-separated values")
        return model_cls(
            num_classes=args.num_classes,
            lr=args.lr,
            dformerv2_pretrained=args.dformerv2_pretrained,
            lambda_mask=args.lambda_mask,
            mask_ratio_depth=args.mask_ratio_depth,
            mask_ratio_primary=args.mask_ratio_primary,
            maskrec_alpha=args.maskrec_alpha,
            maskrec_loss_type=args.maskrec_loss_type,
            maskrec_stage_weights=maskrec_stage_weights,
        )
    if args.model == "dformerv2_cm_infonce":
        contrast_stage_weights = tuple(float(v.strip()) for v in args.contrast_stage_weights.split(","))
        if len(contrast_stage_weights) != 4:
            raise ValueError("--contrast_stage_weights must contain 4 comma-separated values")
        return model_cls(
            num_classes=args.num_classes,
            lr=args.lr,
            dformerv2_pretrained=args.dformerv2_pretrained,
            lambda_contrast=args.lambda_contrast,
            temperature=args.contrast_temperature,
            proj_dim=args.contrast_proj_dim,
            sample_points=args.contrast_sample_points,
            stage_weights=contrast_stage_weights,
        )
    if args.model == "dformerv2_fft_freq_enhance":
        return model_cls(
            num_classes=args.num_classes,
            lr=args.lr,
            dformerv2_pretrained=args.dformerv2_pretrained,
            cutoff_ratio=args.cutoff_ratio,
            gamma_init=args.gamma_init,
        )
    if args.model == "dformerv2_depth_fft_select":
        return model_cls(
            num_classes=args.num_classes,
            lr=args.lr,
            dformerv2_pretrained=args.dformerv2_pretrained,
            cutoff_ratio=args.cutoff_ratio,
        )
    if args.model in {
        "dformerv2_mid_fusion",
    }:
        return model_cls(
            num_classes=args.num_classes,
            lr=args.lr,
            dformerv2_pretrained=args.dformerv2_pretrained,
        )
    return model_cls(num_classes=args.num_classes, lr=args.lr)


def build_callbacks(args, monitor_metric: str):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_callback = DirectStateDictCheckpoint(
        dirpath=args.checkpoint_dir,
        filename_prefix=args.model,
        monitor=monitor_metric,
        mode="max",
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
