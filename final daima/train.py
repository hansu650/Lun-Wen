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
    LitDFormerV2ContextDecoder,
    LitDFormerV2DepthFFTSelect,
    LitDFormerV2FFTFreqEnhance,
    LitDFormerV2FFTHiLoEnhance,
)
from src.models.primkd_lit import LitDFormerV2PrimKD
from src.models.teacher_model import LitDFormerV2GeometryPrimaryTeacher


MODEL_REGISTRY = {
    "early": LitEarlyFusion,
    "mid_fusion": LitMidFusion,
    "dformerv2_mid_fusion": LitDFormerV2MidFusion,
    "dformerv2_context_decoder": LitDFormerV2ContextDecoder,
    "dformerv2_depth_fft_select": LitDFormerV2DepthFFTSelect,
    "dformerv2_fft_freq_enhance": LitDFormerV2FFTFreqEnhance,
    "dformerv2_fft_hilo_enhance": LitDFormerV2FFTHiLoEnhance,
    "dformerv2_geometry_primary_teacher": LitDFormerV2GeometryPrimaryTeacher,
    "dformerv2_primkd_logit_only": LitDFormerV2PrimKD,
}


class DirectStateDictCheckpoint(Callback):
    def __init__(self, dirpath, filename_prefix, monitor, mode="max", save_student_only=False):
        super().__init__()
        self.dirpath = dirpath
        self.filename_prefix = filename_prefix
        self.monitor = monitor
        self.mode = mode
        self.save_student_only = save_student_only
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
        if self.save_student_only and hasattr(pl_module, "export_state_dict"):
            torch.save(pl_module.export_state_dict(), filepath)
        else:
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
    parser.add_argument("--loss_type", type=str, default="ce", choices=["ce", "ce_dice", "dgbf"])
    parser.add_argument("--dice_weight", type=float, default=0.5)
    parser.add_argument("--dgbf_alpha", type=float, default=1.0)
    parser.add_argument("--dgbf_gamma", type=float, default=2.0)
    parser.add_argument(
        "--dgbf_mode",
        type=str,
        default="depth_semantic",
        choices=["depth_semantic", "semantic_only", "depth_only", "focal_only", "none"],
    )
    parser.add_argument("--cutoff_ratio", type=float, default=0.25)
    parser.add_argument("--gamma_init", type=float, default=0.05)
    parser.add_argument("--hilo_alpha_low_init", type=float, default=0.03)
    parser.add_argument("--hilo_alpha_high_init", type=float, default=0.10)
    parser.add_argument("--hilo_alpha_max", type=float, default=0.5)
    parser.add_argument("--hilo_stage_weights", type=str, default="1,1,1,1")
    parser.add_argument("--teacher_ckpt", type=str, default=None)
    parser.add_argument("--kd_weight", type=float, default=0.2)
    parser.add_argument("--kd_temperature", type=float, default=4.0)
    parser.add_argument("--save_student_only", action="store_true")
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
    if args.model == "dformerv2_fft_freq_enhance":
        return model_cls(
            num_classes=args.num_classes,
            lr=args.lr,
            dformerv2_pretrained=args.dformerv2_pretrained,
            cutoff_ratio=args.cutoff_ratio,
            gamma_init=args.gamma_init,
            loss_type=args.loss_type,
            dice_weight=args.dice_weight,
        )
    if args.model == "dformerv2_fft_hilo_enhance":
        hilo_stage_weights = tuple(float(v.strip()) for v in args.hilo_stage_weights.split(","))
        if len(hilo_stage_weights) != 4:
            raise ValueError("--hilo_stage_weights must contain 4 comma-separated values")
        return model_cls(
            num_classes=args.num_classes,
            lr=args.lr,
            dformerv2_pretrained=args.dformerv2_pretrained,
            cutoff_ratio=args.cutoff_ratio,
            alpha_low_init=args.hilo_alpha_low_init,
            alpha_high_init=args.hilo_alpha_high_init,
            alpha_max=args.hilo_alpha_max,
            stage_weights=hilo_stage_weights,
            loss_type=args.loss_type,
            dice_weight=args.dice_weight,
        )
    if args.model == "dformerv2_depth_fft_select":
        return model_cls(
            num_classes=args.num_classes,
            lr=args.lr,
            dformerv2_pretrained=args.dformerv2_pretrained,
            cutoff_ratio=args.cutoff_ratio,
            loss_type=args.loss_type,
            dice_weight=args.dice_weight,
        )
    if args.model == "dformerv2_context_decoder":
        return model_cls(
            num_classes=args.num_classes,
            lr=args.lr,
            dformerv2_pretrained=args.dformerv2_pretrained,
            loss_type=args.loss_type,
            dice_weight=args.dice_weight,
        )
    if args.model == "dformerv2_geometry_primary_teacher":
        return model_cls(
            num_classes=args.num_classes,
            lr=args.lr,
            dformerv2_pretrained=args.dformerv2_pretrained,
            loss_type=args.loss_type,
            dice_weight=args.dice_weight,
            dgbf_alpha=args.dgbf_alpha,
            dgbf_gamma=args.dgbf_gamma,
            dgbf_mode=args.dgbf_mode,
        )
    if args.model == "dformerv2_primkd_logit_only":
        return model_cls(
            num_classes=args.num_classes,
            lr=args.lr,
            dformerv2_pretrained=args.dformerv2_pretrained,
            teacher_ckpt=args.teacher_ckpt,
            kd_weight=args.kd_weight,
            kd_temperature=args.kd_temperature,
            loss_type=args.loss_type,
            dice_weight=args.dice_weight,
        )
    if args.model in {
        "dformerv2_mid_fusion",
    }:
        return model_cls(
            num_classes=args.num_classes,
            lr=args.lr,
            dformerv2_pretrained=args.dformerv2_pretrained,
            loss_type=args.loss_type,
            dice_weight=args.dice_weight,
        )
    return model_cls(
        num_classes=args.num_classes,
        lr=args.lr,
        loss_type=args.loss_type,
        dice_weight=args.dice_weight,
    )


def build_callbacks(args, monitor_metric: str):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_callback = DirectStateDictCheckpoint(
        dirpath=args.checkpoint_dir,
        filename_prefix=args.model,
        monitor=monitor_metric,
        mode="max",
        save_student_only=args.save_student_only,
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
