from pathlib import Path
import shutil

import cv2
import h5py
import numpy as np


OLD_ROOT = Path(r"C:\Users\qintian\Desktop\qintian\data\NYUDepthv2")
MAT_PATH = Path(r"C:\Users\qintian\Desktop\qintian\data\nyu_depth_v2_labeled.mat")
OUT_ROOT = Path(r"C:\Users\qintian\Desktop\qintian\data\NYUDepthv2_matdepth")


def read_split_stems(split_path: Path):
    stems = []
    lines = split_path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        rgb_rel = line.split()[0]
        stems.append(Path(rgb_rel).stem)
    return stems


def copy_base_files():
    if OUT_ROOT.exists():
        raise FileExistsError(f"Output dataset already exists: {OUT_ROOT}")

    OUT_ROOT.mkdir(parents=True)
    shutil.copytree(OLD_ROOT / "RGB", OUT_ROOT / "RGB")
    shutil.copytree(OLD_ROOT / "Label", OUT_ROOT / "Label")
    shutil.copy2(OLD_ROOT / "train.txt", OUT_ROOT / "train.txt")
    shutil.copy2(OLD_ROOT / "test.txt", OUT_ROOT / "test.txt")
    (OUT_ROOT / "Depth").mkdir()


def export_depths():
    with h5py.File(MAT_PATH, "r") as mat_file:
        depths = np.array(mat_file["depths"])

    stems = read_split_stems(OLD_ROOT / "train.txt") + read_split_stems(OLD_ROOT / "test.txt")
    for stem in stems:
        depth_m = depths[int(stem)].T
        depth_mm = np.clip(depth_m * 1000.0, 0, 10000).astype(np.uint16)
        cv2.imwrite(str(OUT_ROOT / "Depth" / f"{stem}.png"), depth_mm)


def print_stats(name, arr):
    print(f"{name}: shape={arr.shape}, dtype={arr.dtype}, min={int(arr.min())}, max={int(arr.max())}")


def sanity_check():
    print("=== NYUDepthv2_matdepth sanity check ===")
    lines = (OUT_ROOT / "train.txt").read_text(encoding="utf-8").splitlines()[:5]
    for idx, line in enumerate(lines):
        rgb_rel, label_rel = line.split()[:2]
        stem = Path(rgb_rel).stem

        rgb = cv2.imread(str(OUT_ROOT / rgb_rel), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(str(OUT_ROOT / "Depth" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        label = cv2.imread(str(OUT_ROOT / label_rel), cv2.IMREAD_UNCHANGED)

        print(f"\n[sample {idx}] stem={stem}")
        print_stats("RGB", rgb)
        print_stats("Depth", depth)
        print_stats("Label", label)
        print(f"Label unique first 50: {np.unique(label)[:50].tolist()}")


def main():
    copy_base_files()
    export_depths()
    sanity_check()
    print(f"\nCreated dataset: {OUT_ROOT}")


if __name__ == "__main__":
    main()
