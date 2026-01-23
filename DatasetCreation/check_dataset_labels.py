"""
Check the number of images with and without labels in combined_dataset2
"""
import os
from pathlib import Path
from collections import defaultdict

# Dataset path
dataset_root = Path("../Combined_Dataset2")

# Check if dataset exists
if not dataset_root.exists():
    print(f"[ERROR] Dataset not found at {dataset_root.absolute()}")
    exit(1)

print(f"[INFO] Analyzing dataset at: {dataset_root.absolute()}")
print("=" * 70)

# Collect statistics
stats = defaultdict(lambda: {'images': 0, 'labeled': 0, 'unlabeled': 0})
total_images = 0
total_labeled = 0
total_unlabeled = 0

# Check train, val, test splits
for split in ['train', 'val', 'test']:
    images_dir = dataset_root / 'images' / split
    labels_dir = dataset_root / 'labels' / split
    
    if not images_dir.exists():
        print(f"[WARNING] {split} images directory not found at {images_dir}")
        continue
    
    print(f"\n{split.upper()} SET:")
    print("-" * 70)
    
    # Count image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    images = [f for f in images_dir.iterdir() if f.suffix.lower() in image_extensions]
    num_images = len(images)
    
    stats[split]['images'] = num_images
    total_images += num_images
    
    # Count images with flowers (non-empty label files)
    num_with_flowers = 0
    num_without_flowers = 0
    
    if labels_dir.exists():
        for img in images:
            label_file = labels_dir / (img.stem + '.txt')
            if label_file.exists() and label_file.stat().st_size > 0:
                num_with_flowers += 1
            else:
                num_without_flowers += 1
    else:
        num_without_flowers = num_images
    
    stats[split]['labeled'] = num_with_flowers
    stats[split]['unlabeled'] = num_without_flowers
    total_labeled += num_with_flowers
    total_unlabeled += num_without_flowers
    
    print(f"  Total images:        {num_images:6d}")
    print(f"  With flowers:        {num_with_flowers:6d} ({100*num_with_flowers/num_images if num_images > 0 else 0:.1f}%)")
    print(f"  Without flowers:     {num_without_flowers:6d} ({100*num_without_flowers/num_images if num_images > 0 else 0:.1f}%)")

# Print summary
print("\n" + "=" * 70)
print("SUMMARY:")
print("=" * 70)
for split in ['train', 'val', 'test']:
    if split in stats and stats[split]['images'] > 0:
        s = stats[split]
        print(f"{split.upper():10} | Images: {s['images']:6d} | With flowers: {s['labeled']:6d} | Without flowers: {s['unlabeled']:6d}")

print("-" * 70)
print(f"{'TOTAL':10} | Images: {total_images:6d} | With flowers: {total_labeled:6d} | Without flowers: {total_unlabeled:6d}")
print("=" * 70)
