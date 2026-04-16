"""统一训练脚本"""
import os # 导入 Python 标准库 os，后面要用它改环境变量、创建目录。
import argparse # 导入命令行参数库。你运行 python train.py --help 时，显示帮助信息就是它做的。
import warnings # 导入警告处理库，后面用来屏蔽一些已知但不影响运行的 warning。

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")# 给环境变量设默认值。
os.environ.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")
warnings.filterwarnings(
    "ignore",
    message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
    module=r"lightning\.pytorch\.utilities\._pytree",
) # 忽略warnings

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
# 导入回调，自动保存最优模型和如果指标长期不变好就提前终止训练
from src.data_module import NYUDataModule # 导入数据模块，里面封装了数据加载、增强等逻辑。
from src.models.advanced_lit_module import LitAdvancedRGBD
from src.models.attention_fusion_model import LitAttentionFusion
from src.models.dformer_model import LitDFormerInspired
from src.models.early_fusion import LitEarlyFusion # 导入 Early Fusion 模型。
from src.models.mid_fusion import LitMidFusion # 导入 Mid Fusion 模型。


MODEL_REGISTRY = {
    "early": LitEarlyFusion,
    "mid_fusion": LitMidFusion,
    "attention": LitAttentionFusion,
    "advanced": LitAdvancedRGBD,
    "dformer": LitDFormerInspired,
}# 用字符串代替模型类名

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
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--min_lr_ratio", type=float, default=0.05)
    parser.add_argument("--backbone_lr_mult", type=float, default=0.1)
    parser.add_argument("--eval_tta", type=parse_bool_flag, default=False)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    return parser
# 参数默认定义

def parse_devices(devices: str):
    if devices == "auto":
        return "auto"
    try:
        return int(devices)
    except ValueError:
        return devices
# 整理cpu gpu

def build_datamodule(args):
    return NYUDataModule(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
# 定义成函数就可以在main函数中调用

def build_model(args):
    model_cls = MODEL_REGISTRY[args.model]
    if args.model == "advanced":
        return model_cls(
            num_classes=args.num_classes,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
            backbone_lr_mult=args.backbone_lr_mult,
            eval_tta=args.eval_tta,
        )
    return model_cls(num_classes=args.num_classes, lr=args.lr)
# 定义成函数就可以在main函数中调用 命令行传入参数

def build_callbacks(args, monitor_metric: str): # 处理回调
    os.makedirs(args.checkpoint_dir, exist_ok=True) # 保存模型的目录
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename=f"{args.model}" + "-{epoch:02d}-{" + monitor_metric + ":.4f}",
        monitor=monitor_metric, # 监控指标
        mode="max", # 指标越大越好
        save_top_k=1, # 保存一个
    )
    early_stop_callback = EarlyStopping(
        monitor=monitor_metric,
        patience=args.early_stop_patience, # 早停策略
        mode="max",
    )# 一样的指标
    return checkpoint_callback, early_stop_callback


def build_trainer(args, callbacks):
    return L.Trainer(
        max_epochs=args.max_epochs, # 轮数
        accelerator=args.accelerator, # 加速器
        devices=parse_devices(args.devices), # 设备
        callbacks=list(callbacks), # 回调
        default_root_dir=args.checkpoint_dir, # 保存模型的目录
        log_every_n_steps=10, # 每10步记录一次日志
        check_val_every_n_epoch=args.check_val_every_n_epoch, # 长训练时可以少做几次验证
    )


def main():
    args = build_parser().parse_args() # 解析参数 把命令行里面的参数解析成python可以用的的对象
    monitor_metric = "val/mIoU"
    datamodule = build_datamodule(args) # 创建数据模块
    model = build_model(args) # 创建模型
    checkpoint_callback, early_stop_callback = build_callbacks(args, monitor_metric) # 创建回调
    trainer = build_trainer(args, callbacks=[checkpoint_callback, early_stop_callback]) # 创建训练器
    print(f"开始训练模型: {args.model}")
    trainer.fit(model, datamodule=datamodule)# 训练 lightning会自动调用一些函数(上面我们指定的参数的这些函数)
    best_score = checkpoint_callback.best_model_score
    best_score_text = "N/A" if best_score is None else f"{best_score:.4f}"
    print(f"训练完成！最优模型: {checkpoint_callback.best_model_path}")
    print(f"最优 {monitor_metric}: {best_score_text}")


if __name__ == "__main__":
    main()
# 脚本入口保护，只有直接运行才会执行不然import不会执行
