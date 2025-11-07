import os
import shutil
from pathlib import Path

# Paths to your two datasets
dataset1 = Path("../Strawberry_Flower_Dataset2")  # has train/, valid/, test/
dataset2 = Path("../Strawberry_Flower_Dataset")   # has images/train,val + labels/train,val
combined = Path("../Combined_Dataset")

# Create target folders
for split in ["train", "val", "test"]:
    (combined / "images" / split).mkdir(parents=True, exist_ok=True)
    (combined / "labels" / split).mkdir(parents=True, exist_ok=True)

def copy_data(img_dir, lbl_dir, split_name):
    """Copy all image/label pairs from given directories to combined dataset."""
    img_dir = Path(img_dir)
    lbl_dir = Path(lbl_dir)

    for img_path in img_dir.glob("*.*"):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        label_path = lbl_dir / (img_path.stem + ".txt")

        dst_img = combined / "images" / split_name / img_path.name
        dst_lbl = combined / "labels" / split_name / (img_path.stem + ".txt")

        # Rename if filename collision occurs
        if dst_img.exists():
            dst_img = dst_img.with_stem(dst_img.stem + "_dup")
            dst_lbl = dst_lbl.with_stem(dst_lbl.stem + "_dup")

        shutil.copy2(img_path, dst_img)
        if label_path.exists():
            shutil.copy2(label_path, dst_lbl)
        else:
            print(f"⚠️ No label found for {img_path.name}")

# --- Dataset 1 (train/valid/test) ---
for split in ["train", "valid", "test"]:
    img_dir = dataset1 / split / "images"
    lbl_dir = dataset1 / split / "labels"
    if img_dir.exists() and lbl_dir.exists():
        new_split_name = "val" if split == "valid" else split
        copy_data(img_dir, lbl_dir, new_split_name)

# --- Dataset 2 (images/labels subfolders) ---
for split in ["train", "val"]:
    img_dir = dataset2 / "images" / split
    lbl_dir = dataset2 / "labels" / split
    if img_dir.exists() and lbl_dir.exists():
        copy_data(img_dir, lbl_dir, split)

print(f"✅ Combined dataset created at: {combined}")
