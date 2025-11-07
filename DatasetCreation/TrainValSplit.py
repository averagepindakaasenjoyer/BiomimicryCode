import os
import xml.etree.ElementTree as ET
import dotenv
import random, shutil


dotenv.load_dotenv()

images_dir = os.getenv("IMAGES_DIR")
annotations_dir = os.getenv("ANNOTATIONS_DIR")
labels_dir = os.getenv("LABELS_DIR")

print(f"Images directory: {images_dir}")
print(f"Annotations directory: {annotations_dir}")
print(f"Labels directory: {labels_dir}")


# Ensure target subdirectories exist before moving files
for sub in ("train", "val"):
    imgs_sub = os.path.join(images_dir, sub)
    labs_sub = os.path.join(labels_dir, sub)
    os.makedirs(imgs_sub, exist_ok=True)
    os.makedirs(labs_sub, exist_ok=True)

images = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg",".png"))]
random.shuffle(images)
split = int(0.8 * len(images))
train = images[:split]
val = images[split:]

for name in train:
    src_img = os.path.join(images_dir, name)
    dst_img = os.path.join(images_dir, "train", name)
    if os.path.exists(src_img):
        try:
            shutil.move(src_img, dst_img)
        except Exception as e:
            print(f"Error moving image {src_img} -> {dst_img}: {e}")
    else:
        print(f"Warning: image not found, skipping: {src_img}")

    lbl = os.path.splitext(name)[0] + ".txt"
    src_lbl = os.path.join(labels_dir, lbl)
    dst_lbl = os.path.join(labels_dir, "train", lbl)
    if os.path.exists(src_lbl):
        try:
            shutil.move(src_lbl, dst_lbl)
        except Exception as e:
            print(f"Error moving label {src_lbl} -> {dst_lbl}: {e}")
    else:
        print(f"Warning: label file not found for {name}, skipping: {src_lbl}")

for name in val:
    src_img = os.path.join(images_dir, name)
    dst_img = os.path.join(images_dir, "val", name)
    if os.path.exists(src_img):
        try:
            shutil.move(src_img, dst_img)
        except Exception as e:
            print(f"Error moving image {src_img} -> {dst_img}: {e}")
    else:
        print(f"Warning: image not found, skipping: {src_img}")

    lbl = os.path.splitext(name)[0] + ".txt"
    src_lbl = os.path.join(labels_dir, lbl)
    dst_lbl = os.path.join(labels_dir, "val", lbl)
    if os.path.exists(src_lbl):
        try:
            shutil.move(src_lbl, dst_lbl)
        except Exception as e:
            print(f"Error moving label {src_lbl} -> {dst_lbl}: {e}")
    else:
        print(f"Warning: label file not found for {name}, skipping: {src_lbl}")