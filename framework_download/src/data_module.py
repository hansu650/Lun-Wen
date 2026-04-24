import os
from typing import Callable, List, Optional, Tuple

import albumentations as A
import cv2
import lightning as L
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset


class NYUDataset(Dataset):
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
        with open(file_path, "r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle.readlines() if line.strip()]

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
        # 当前 depth-only 主线读取原始 Depth/。
        depth_path = os.path.join(self.data_root, "Depth", f"{stem}.png")
        label_path = os.path.join(self.data_root, gt_rel)

        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if rgb is None:
            raise FileNotFoundError(f"Missing RGB image: {rgb_path}")
        if depth is None:
            raise FileNotFoundError(f"Missing depth image: {depth_path}")
        if label is None:
            raise FileNotFoundError(f"Missing label image: {label_path}")

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            rgb, depth, label = self.transform(rgb, depth, label)

        return {"rgb": rgb, "depth": depth, "label": label}


def get_train_transform(image_size=(480, 640)):
    h, w = image_size
    return A.Compose(
        [
            A.RandomCrop(height=h, width=w, p=1.0) if h < 480 or w < 640 else A.NoOp(),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        additional_targets={"depth": "image", "label": "mask"},
    )


def get_val_transform():
    return A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        additional_targets={"depth": "image", "label": "mask"},
    )


class AlbumentationsTransform:
    def __init__(self, transform: A.Compose):
        self.transform = transform

    def __call__(self, rgb, depth, label):
        result = self.transform(image=rgb, depth=depth, label=label)
        rgb_tensor = result["image"]
        depth_tensor = result["depth"]
        label_tensor = result["label"]

        if not torch.is_tensor(depth_tensor):
            depth_tensor = torch.from_numpy(depth_tensor).float()
        if depth_tensor.dim() == 2:
            depth_tensor = depth_tensor.unsqueeze(0)
        elif depth_tensor.dim() == 3 and depth_tensor.shape[0] == 1:
            pass
        else:
            raise ValueError(f"Depth tensor should be [H,W] or [1,H,W], got {tuple(depth_tensor.shape)}")
        depth_tensor = depth_tensor.float() / 255.0

        if not torch.is_tensor(label_tensor):
            label_tensor = torch.from_numpy(label_tensor)
        if label_tensor.dim() == 3 and label_tensor.shape[0] == 1:
            label_tensor = label_tensor.squeeze(0)
        label_tensor = label_tensor.long()

        return rgb_tensor, depth_tensor, label_tensor


class NYUDataModule(L.LightningDataModule):
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
            self.data_root,
            split="train",
            transform=train_transform,
            image_size=self.image_size,
        )
        self.val_dataset = NYUDataset(
            self.data_root,
            split="test",
            transform=val_transform,
            image_size=self.image_size,
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
