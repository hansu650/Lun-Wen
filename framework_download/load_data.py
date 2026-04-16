import os
import cv2

# 检查路径
data_root = "data/NYUDepthv2"
for item in ["RGB", "Depth", "Label", "train.txt", "test.txt"]:
    path = os.path.join(data_root, item)
    print(f"{'✅' if os.path.exists(path) else '❌'} {path}")

# 读取样例
rgb = cv2.imread("../data/NYUDepthv2/RGB/0.jpg")
depth = cv2.imread("../data/NYUDepthv2/Depth/0.png", cv2.IMREAD_GRAYSCALE)
label = cv2.imread("../data/NYUDepthv2/Label/0.png", cv2.IMREAD_GRAYSCALE)

print(f"\nRGB 形状: {rgb.shape}")      # (480, 640, 3)
print(f"Depth 形状: {depth.shape}")   # (480, 640)
print(f"Label 形状: {label.shape}")   # (480, 640)