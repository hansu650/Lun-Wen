"""可视化工具函数"""
import numpy as np
import torch
import cv2

HAS_MATPLOTLIB = False
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    pass


NYU_COLORMAP = np.array([
    [0, 0, 0], [174, 199, 232], [152, 223, 138], [31, 119, 180], [255, 152, 150],
    [148, 103, 189], [197, 176, 213], [140, 86, 75], [196, 156, 148], [227, 119, 194],
    [247, 182, 210], [127, 127, 127], [199, 199, 199], [188, 189, 34], [219, 219, 141],
    [23, 190, 207], [158, 218, 229], [255, 127, 14], [255, 187, 120], [44, 160, 44],
    [214, 39, 40], [255, 69, 0], [0, 128, 128], [255, 215, 0], [75, 0, 130],
    [240, 230, 140], [124, 252, 0], [70, 130, 180], [255, 20, 147], [210, 105, 30],
    [0, 255, 255], [128, 128, 0], [255, 165, 0], [176, 224, 230], [255, 192, 203],
    [221, 160, 221], [128, 0, 128], [173, 255, 47], [135, 206, 250], [250, 128, 114],
    [240, 128, 128]
], dtype=np.uint8)


def denormalize_rgb(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """反归一化 RGB 张量"""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    img = tensor * std + mean
    img = torch.clamp(img, 0, 1)
    return img.permute(1, 2, 0).numpy()


def label_to_color(label):
    """将语义标签转换为彩色图"""
    color = NYU_COLORMAP[label % 40]
    return color


def visualize_prediction(rgb, pred, gt=None, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """可视化预测结果"""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is not available.")
    
    if torch.is_tensor(rgb):
        rgb = denormalize_rgb(rgb, mean, std)
    if torch.is_tensor(pred):
        pred = pred.numpy()
    if gt is not None and torch.is_tensor(gt):
        gt = gt.numpy()
    
    n_cols = 3 if gt is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 4, 4))
    
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[0].axis("off")
    
    axes[1].imshow(label_to_color(pred))
    axes[1].set_title("Prediction")
    axes[1].axis("off")
    
    if gt is not None:
        axes[2].imshow(label_to_color(gt))
        axes[2].set_title("Ground Truth")
        axes[2].axis("off")
    
    plt.tight_layout()
    return fig
