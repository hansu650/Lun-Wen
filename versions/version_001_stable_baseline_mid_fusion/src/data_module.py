"""NYU Depth V2 数据模块"""
import os # 拼路径
from typing import Tuple, Optional, Callable, List # 导入类型

import cv2 # 读取图片
import numpy as np
import torch # 张量
from torch.utils.data import Dataset, DataLoader
import lightning as L
import albumentations as A  #图像增强库
from albumentations.pytorch import ToTensorV2 # 将图片转换为张量


class NYUDataset(Dataset):
    """NYU Depth V2 Dataset for RGB-D Semantic Segmentation"""

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        image_size: Tuple[int, int] = (480, 640), # 图像大小
    ):
        self.data_root = data_root # 数据根目录
        self.split = split # 数据集划分
        self.transform = transform # 数据增强
        self.image_size = image_size # 图像大小

        split_file = os.path.join(data_root, "train.txt" if split == "train" else "test.txt")
        self.samples = self._read_split_file(split_file) # 先把划分的txt文件读一下
        print(f"[NYUDataset] {split}: {len(self.samples)} samples")
    # 读取划分的txt文件,得到相对路径(图片,标签)
    def _read_split_file(self, file_path: str) -> List[Tuple[str, str]]:
        pairs = []
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                rgb_rel, gt_rel = parts[0], parts[1]
            else:
                rgb_rel = parts[0]
                stem = os.path.splitext(os.path.basename(rgb_rel))[0]
                gt_rel = f"Label/{stem}.png"
            pairs.append((rgb_rel, gt_rel))
        return pairs
    # 返回数据集长度
    def __len__(self) -> int:
        return len(self.samples)
    # 单个样本怎么读
    def __getitem__(self, idx: int):
        rgb_rel, gt_rel = self.samples[idx]
        stem = os.path.splitext(os.path.basename(rgb_rel))[0]# 找到相对路径
        # 拼路径
        rgb_path = os.path.join(self.data_root, rgb_rel)
        gt_path = os.path.join(self.data_root, gt_rel)
        depth_path = os.path.join(self.data_root, "Depth", f"{stem}.png")

        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB) # 转换为RGB通道格式
        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE) # 读取深度
        label = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) # 读取标签

        if self.transform is not None: # 如果有数据增强
            rgb, depth, label = self.transform(rgb, depth, label) # 数据增强

        return {"rgb": rgb, "depth": depth, "label": label}


def get_train_transform(image_size=(480, 640)):
    """训练集数据增强"""
    h, w = image_size
    return A.Compose([
        A.RandomCrop(height=h, width=w, p=1.0) if h < 480 or w < 640 else A.NoOp(),
        A.HorizontalFlip(p=0.5), # 可能左右翻转
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),# 标准化 ImageNet配置
        ToTensorV2(),
    ], additional_targets={"depth": "image", "label": "mask"})
# depth可以和image一样，label不做归一化破坏数值的操作

def get_val_transform():
    """验证集数据增强"""
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(), # 验证时只做确定的预处理
    ], additional_targets={"depth": "image", "label": "mask"})


class AlbumentationsTransform:
    """
    可序列化的 albumentations 变换包装器（解决 Windows 多进程 pickle 问题）
    """
    def __init__(self, transform: A.Compose):
        self.transform = transform

    def __call__(self, rgb, depth, label):
        result = self.transform(image=rgb, depth=depth, label=label)
        rgb_tensor = result["image"]
        depth_tensor = result["depth"]
        label_tensor = result["label"]

        # albumentations 的 ToTensorV2 会将 image 转为 [C,H,W]，但 depth 和 mask 需要手动处理
        if not torch.is_tensor(depth_tensor):
            depth_tensor = torch.from_numpy(depth_tensor).float()
        if depth_tensor.dim() == 2:
            depth_tensor = depth_tensor.unsqueeze(0) / 255.0
        elif depth_tensor.dim() == 3 and depth_tensor.shape[0] == 3:
            depth_tensor = depth_tensor[0:1, :, :] / 255.0

        if not torch.is_tensor(label_tensor):
            label_tensor = torch.from_numpy(label_tensor)
        if label_tensor.dim() == 3 and label_tensor.shape[0] == 1:
            label_tensor = label_tensor.squeeze(0)
        label_tensor = label_tensor.long()# 标签变成整形

        return rgb_tensor, depth_tensor, label_tensor

# 统一组织
class NYUDataModule(L.LightningDataModule):
    """PyTorch Lightning 数据模块"""

    def __init__(
        self,
        data_root: str,
        batch_size: int = 4,
        num_workers: int = 4,
        image_size: Tuple[int, int] = (480, 640),
    ):
        super().__init__()
        self.save_hyperparameters()# 记录一下参数
        self.data_root = data_root# 参数变成属性
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

    def setup(self, stage: Optional[str] = None): # 实例化
        train_transform = AlbumentationsTransform(get_train_transform(self.image_size))
        val_transform = AlbumentationsTransform(get_val_transform())

        self.train_dataset = NYUDataset( # 训练集
            self.data_root, split="train", transform=train_transform, image_size=self.image_size
        )
        self.val_dataset = NYUDataset( # 验证集
            self.data_root, split="test", transform=val_transform, image_size=self.image_size
        )
    # 实例化
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
    
    def test_dataloader(self):
        return self.val_dataloader()
