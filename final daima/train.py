"""Training entry point for active RGB-D segmentation experiments."""
import argparse
import os
import sys
import warnings

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
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
from lightning.pytorch.callbacks import Callback, EarlyStopping, TQDMProgressBar

from src.data_module import NYUDataModule
from src.models.early_fusion import LitEarlyFusion
from src.models.mid_fusion import (
    LitDFormerV2BranchDepthAdapter,
    LitDFormerV2BranchDepthBlendAdapter,
    LitDFormerV2DepthEncoderBNEval,
    LitDFormerV2HamDecoder,
    LitDFormerV2MidFusion,
    LitMidFusion,
)
from src.models.primkd_lit import LitDFormerV2PrimKD
from src.models.teacher_model import (
    LitDFormerV2GeometryPrimaryHamDecoder,
    LitDFormerV2GeometryPrimaryTeacher,
)
from src.models.tgga_adapter import LitDFormerV2TGGAC4OnlyBeta002Aux003DetachSemSimpleFPNV1


ACTIVE_MODEL_REGISTRY = {
    "dformerv2_mid_fusion": LitDFormerV2MidFusion,
    "dformerv2_ham_decoder": LitDFormerV2HamDecoder,
    "dformerv2_branch_depth_adapter": LitDFormerV2BranchDepthAdapter,
    "dformerv2_branch_depth_blend_adapter": LitDFormerV2BranchDepthBlendAdapter,
    "dformerv2_depth_encoder_bn_eval": LitDFormerV2DepthEncoderBNEval,
    "dformerv2_tgga_c4only_beta002_aux003_detachsem_simplefpn_v1": LitDFormerV2TGGAC4OnlyBeta002Aux003DetachSemSimpleFPNV1,
    "dformerv2_geometry_primary_ham_decoder": LitDFormerV2GeometryPrimaryHamDecoder,
    "dformerv2_geometry_primary_teacher": LitDFormerV2GeometryPrimaryTeacher,
    "dformerv2_primkd_logit_only": LitDFormerV2PrimKD,
}

LEGACY_MODEL_REGISTRY = {
    "early": LitEarlyFusion,
    "mid_fusion": LitMidFusion,
}

MODEL_REGISTRY = {
    **ACTIVE_MODEL_REGISTRY,
    **LEGACY_MODEL_REGISTRY,
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
    parser = argparse.ArgumentParser(description="RGB-D semantic segmentation training")
    parser.add_argument("--model", type=str, default="dformerv2_mid_fusion", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--data_root", type=str, required=True, help="NYU Depth V2 dataset root")
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
    parser.add_argument("--loss_type", type=str, default="ce", choices=["ce"])
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
    if args.model == "dformerv2_primkd_logit_only":
        return model_cls(
            num_classes=args.num_classes,
            lr=args.lr,
            dformerv2_pretrained=args.dformerv2_pretrained,
            teacher_ckpt=args.teacher_ckpt,
            kd_weight=args.kd_weight,
            kd_temperature=args.kd_temperature,
            loss_type=args.loss_type,
        )
    if args.model in {
        "dformerv2_mid_fusion",
        "dformerv2_ham_decoder",
        "dformerv2_branch_depth_adapter",
        "dformerv2_branch_depth_blend_adapter",
        "dformerv2_depth_encoder_bn_eval",
        "dformerv2_tgga_c4only_beta002_aux003_detachsem_simplefpn_v1",
        "dformerv2_geometry_primary_ham_decoder",
        "dformerv2_geometry_primary_teacher",
    }:
        return model_cls(
            num_classes=args.num_classes,
            lr=args.lr,
            dformerv2_pretrained=args.dformerv2_pretrained,
            loss_type=args.loss_type,
        )
    return model_cls(
        num_classes=args.num_classes,
        lr=args.lr,
        loss_type=args.loss_type,
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
    progress_callback = TQDMProgressBar(refresh_rate=10)
    return checkpoint_callback, early_stop_callback, progress_callback


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
    checkpoint_callback, early_stop_callback, progress_callback = build_callbacks(args, monitor_metric)
    trainer = build_trainer(args, callbacks=[checkpoint_callback, early_stop_callback, progress_callback])
    print(f"Starting training model: {args.model}")
    trainer.fit(model, datamodule=datamodule)
    best_score = checkpoint_callback.best_model_score
    best_score_text = "N/A" if best_score is None else f"{best_score:.4f}"
    print(f"Training complete. Best model: {checkpoint_callback.best_model_path}")
    print(f"Best {monitor_metric}: {best_score_text}")


if __name__ == "__main__":
    main()
