"""NYU Depth V2 数据模块"""
import os
from typing import Tuple, Optional, Callable, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
import albumentations as A
from albumentations.pytorch import ToTensorV2


DFORMER_DEPTH_MEAN = 0.48
DFORMER_DEPTH_STD = 0.28


def map_nyu40_labels_to_train_ids(label_tensor: torch.Tensor) -> torch.Tensor:
    label_tensor = label_tensor.long()
    ignore_mask = label_tensor == 0
    label_tensor = label_tensor - 1
    label_tensor = label_tensor.masked_fill(ignore_mask, 255)
    invalid = (label_tensor < 0) | ((label_tensor >= 40) & (label_tensor != 255))
    return label_tensor.masked_fill(invalid, 255)


def normalize_nyu_depth_to_dformer(depth_tensor: torch.Tensor) -> torch.Tensor:
    depth_tensor = depth_tensor.float()
    if depth_tensor.dim() == 3 and depth_tensor.shape[0] == 1:
        depth_tensor = depth_tensor.squeeze(0)
    depth_tensor = depth_tensor.unsqueeze(0) / 255.0
    return (depth_tensor - DFORMER_DEPTH_MEAN) / DFORMER_DEPTH_STD


class NYUDataset(Dataset):
    """NYU Depth V2 Dataset for RGB-D Semantic Segmentation"""
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        image_size: Tuple[int, int] = (480, 640),
    ):
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.image_size = image_size
        
        split_file = os.path.join(data_root, "train.txt" if split == "train" else "test.txt")
        self.samples = self._read_split_file(split_file)
        print(f"[NYUDataset] {split}: {len(self.samples)} samples")
    
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
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        rgb_rel, gt_rel = self.samples[idx]
        stem = os.path.splitext(os.path.basename(rgb_rel))[0]
        
        rgb_path = os.path.join(self.data_root, rgb_rel)
        gt_path = os.path.join(self.data_root, gt_rel)
        depth_path = os.path.join(self.data_root, "Depth", f"{stem}.png")
        
        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        if self.transform is not None:
            rgb, depth, label = self.transform(rgb, depth, label)
        
        return {"rgb": rgb, "depth": depth, "label": label}


def get_train_transform(image_size=(480, 640)):
    """训练集数据增强"""
    h, w = image_size
    return A.Compose([
        A.RandomCrop(height=h, width=w, p=1.0) if h < 480 or w < 640 else A.NoOp(),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], additional_targets={"depth": "mask", "label": "mask"})


def get_val_transform():
    """验证集数据增强"""
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], additional_targets={"depth": "mask", "label": "mask"})


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
            depth_tensor = torch.from_numpy(depth_tensor)
        depth_tensor = normalize_nyu_depth_to_dformer(depth_tensor)
            
        if not torch.is_tensor(label_tensor):
            label_tensor = torch.from_numpy(label_tensor)
        if label_tensor.dim() == 3 and label_tensor.shape[0] == 1:
            label_tensor = label_tensor.squeeze(0)
        label_tensor = map_nyu40_labels_to_train_ids(label_tensor)
        
        return rgb_tensor, depth_tensor, label_tensor


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
        self.save_hyperparameters()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
    
    def setup(self, stage: Optional[str] = None):
        train_transform = AlbumentationsTransform(get_train_transform(self.image_size))
        val_transform = AlbumentationsTransform(get_val_transform())
        
        self.train_dataset = NYUDataset(
            self.data_root, split="train", transform=train_transform, image_size=self.image_size
        )
        self.val_dataset = NYUDataset(
            self.data_root, split="test", transform=val_transform, image_size=self.image_size
        )
    
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
