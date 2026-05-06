from pathlib import Path

import cv2
import h5py
import numpy as np


DATA_ROOT = Path(r"C:\Users\qintian\Desktop\qintian\data\NYUDepthv2")
MAT_PATH = Path(r"C:\Users\qintian\Desktop\qintian\data\nyu_depth_v2_labeled.mat")
PREVIEW_DIR = Path(r"C:\Users\qintian\Desktop\qintian\framework_download\data_check_preview")


def print_array_stats(name, arr):
    print(f"{name}:")
    print(f"  shape: {arr.shape}")
    print(f"  dtype: {arr.dtype}")
    print(f"  min/max: {float(np.min(arr)):.6f} / {float(np.max(arr)):.6f}")


def save_gray_preview(path, arr):
    arr = arr.astype(np.float32)
    mn = float(arr.min())
    mx = float(arr.max())
    preview = ((arr - mn) / (mx - mn) * 255.0).astype(np.uint8)
    cv2.imwrite(str(path), preview)


def save_label_preview(path, label):
    label = label.astype(np.uint8)
    preview = cv2.applyColorMap((label * 6).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(str(path), preview)


def load_folder_samples():
    train_lines = (DATA_ROOT / "train.txt").read_text(encoding="utf-8").splitlines()
    return [line.split()[:2] for line in train_lines[:5]]


def check_folder():
    print("=== Folder samples ===")
    samples = load_folder_samples()
    for idx, (rgb_rel, label_rel) in enumerate(samples):
        stem = Path(rgb_rel).stem
        rgb_path = DATA_ROOT / rgb_rel
        depth_path = DATA_ROOT / "Depth" / f"{stem}.png"
        label_path = DATA_ROOT / label_rel

        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_UNCHANGED)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)

        print(f"\n[folder sample {idx}]")
        print(f"RGB path: {rgb_path}")
        print_array_stats("RGB", rgb)
        print(f"Depth path: {depth_path}")
        print_array_stats("Depth", depth)
        depth_kind = "RGBA" if depth.ndim == 3 and depth.shape[2] == 4 else "grayscale"
        print(f"  depth kind: {depth_kind}")
        print(f"Label path: {label_path}")
        print_array_stats("Label", label)
        label_unique = np.unique(label)
        print(f"  unique first 50: {label_unique[:50].tolist()}")
        print(f"  contains 0: {bool(np.any(label_unique == 0))}")
        print(f"  contains 40: {bool(np.any(label_unique == 40))}")

        rgb_save = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb.ndim == 3 and rgb.shape[2] == 3 else rgb
        cv2.imwrite(str(PREVIEW_DIR / f"folder_rgb_{idx}.png"), rgb_save)
        save_gray_preview(PREVIEW_DIR / f"folder_depth_{idx}.png", depth[..., 0] if depth.ndim == 3 else depth)
        save_label_preview(PREVIEW_DIR / f"folder_label_{idx}.png", label[..., 0] if label.ndim == 3 else label)


def read_mat_var(mat_file, name):
    data = mat_file[name][()]
    return np.array(data)


def get_mat_sample(arr, idx):
    if arr.ndim == 4:
        sample = arr[idx]
        if sample.shape[0] in (1, 3):
            sample = np.transpose(sample, (1, 2, 0))
        return sample
    if arr.ndim == 3:
        return arr[idx]
    raise ValueError(f"Unexpected mat sample ndim for shape {arr.shape}")


def check_mat():
    print("\n=== MAT variables ===")
    with h5py.File(MAT_PATH, "r") as mat_file:
        images = read_mat_var(mat_file, "images")
        depths = read_mat_var(mat_file, "depths")
        raw_depths = read_mat_var(mat_file, "rawDepths")
        labels = read_mat_var(mat_file, "labels")

    for name, arr in [
        ("images", images),
        ("depths", depths),
        ("rawDepths", raw_depths),
        ("labels", labels),
    ]:
        print_array_stats(name, arr)
        if name == "labels":
            label_unique = np.unique(arr)
            print(f"  labels unique first 100: {label_unique[:100].tolist()}")
            print(f"  labels max > 40: {bool(label_unique.max() > 40)}")

    if int(labels.max()) <= 40:
        print("MAT label semantic: possible 40 classes; try 0=ignore, 1~40 -> 0~39.")
    else:
        print("MAT label semantic: not direct 40-class labels; NYU40 mapping is needed.")

    for idx in range(5):
        mat_rgb = get_mat_sample(images, idx)
        mat_depth = get_mat_sample(depths, idx)
        mat_label = get_mat_sample(labels, idx)

        if mat_rgb.shape[-1] == 3:
            mat_rgb_save = cv2.cvtColor(mat_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
        else:
            mat_rgb_save = mat_rgb.astype(np.uint8)
        cv2.imwrite(str(PREVIEW_DIR / f"mat_rgb_{idx}.png"), mat_rgb_save)
        save_gray_preview(PREVIEW_DIR / f"mat_depth_{idx}.png", mat_depth)
        save_label_preview(PREVIEW_DIR / f"mat_label_{idx}.png", mat_label)


def main():
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    check_folder()
    check_mat()
    print(f"\nPreview saved to: {PREVIEW_DIR}")


if __name__ == "__main__":
    main()
