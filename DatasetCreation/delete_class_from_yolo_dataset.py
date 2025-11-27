#!/usr/bin/env python3
r"""
delete_class_from_yolo_dataset.py

Safe utility to remove a class from a YOLO-style dataset (text label files).
- Backs up the labels folder before modifying it
- Removes all label lines referencing the class (by name or id)
- Remaps remaining class ids to contiguous 0..nc-1
- Optionally moves images that end up with no labels into labels/removed_images
- Optionally updates data.yaml (nc and names)

Usage examples (PowerShell):
  python .\DatasetCreation\delete_class_from_yolo_dataset.py --labels-dir ..\Combined_Dataset\labels --images-dir ..\Combined_Dataset\images --data-yaml ..\data.yaml --remove-class "flower-bud" --remove-empty-images
  python .\DatasetCreation\delete_class_from_yolo_dataset.py --labels-dir ..\Combined_Dataset\labels --remove-id 1

Be sure to backup your dataset before running this on important data.
"""

import argparse
import os
import shutil
import time
import glob

try:
    import yaml
except Exception:
    yaml = None

import dotenv

dotenv.load_dotenv()


def backup_dir(path):
    ts = time.strftime("%Y%m%dT%H%M%S")
    dst = f"{path}.bak_{ts}"
    shutil.copytree(path, dst)
    return dst


def load_data_yaml(path):
    if path and os.path.exists(path):
        if yaml is None:
            raise RuntimeError("PyYAML is required to read data.yaml. Install with: pip install pyyaml")
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return None


def save_data_yaml(path, data):
    if yaml is None:
        raise RuntimeError("PyYAML is required to write data.yaml. Install with: pip install pyyaml")
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False)



def run(labels_dir,
        images_dir=None,
        data_yaml_path=None,
        remove_by='name',  # 'name' or 'id'
        remove_class_name=None,
        remove_class_id=None,
        remove_empty_images=False,
        do_backup=True,
        update_data_yaml=True,
        preview_only=True):
    """Run the delete-class operation with explicit parameters.

    If preview_only is True the function will only print a summary of changes and not write files.
    """

    if not os.path.isdir(labels_dir):
        raise SystemExit(f"Labels dir not found: {labels_dir}")

    data = load_data_yaml(data_yaml_path) if data_yaml_path else None
    name_to_id = {}
    if data and 'names' in data:
        names = data['names']
        # normalize to list
        if isinstance(names, dict):
            idx_names = [None] * (max(int(k) for k in names.keys()) + 1)
            for k, v in names.items():
                idx_names[int(k)] = v
            names = idx_names
        if not isinstance(names, list):
            raise SystemExit("Unsupported names format in data.yaml")
        for i, n in enumerate(names):
            name_to_id[str(n)] = i

    # determine remove ids
    remove_ids = set()
    if remove_by == 'id':
        if remove_class_id is None:
            raise SystemExit("remove_by='id' but remove_class_id is None")
        remove_ids.add(int(remove_class_id))
    else:
        if data is None:
            raise SystemExit("data.yaml must be provided to remove by class name")
        if remove_class_name is None:
            raise SystemExit("remove_by='name' but remove_class_name is None")
        if remove_class_name not in name_to_id:
            raise SystemExit(f"Class name '{remove_class_name}' not found in data.yaml names: {name_to_id}")
        remove_ids.add(name_to_id[remove_class_name])

    print("Will remove class ids:", remove_ids)

    # collect all label files (recursively). This finds labels under splits like train/val as well.
    pattern = os.path.join(labels_dir, '**', '*.txt')
    txts = [os.path.relpath(p, labels_dir) for p in glob.glob(pattern, recursive=True)]
    removed_images_dir = None
    if remove_empty_images and images_dir:
        removed_images_dir = os.path.join(labels_dir, "removed_images")
        if not preview_only:
            os.makedirs(removed_images_dir, exist_ok=True)

    # pass 1: compute remaining class ids present and preview counts
    remaining_ids_set = set()
    files_info = {}
    total_removed_lines = 0
    total_lines = 0
    for t in txts:
        path = os.path.join(labels_dir, t)
        try:
            lines = [ln.strip() for ln in open(path, 'r', encoding='utf-8').read().splitlines() if ln.strip()]
        except Exception:
            lines = []
        # ignore extraneous lines
        kept = []
        removed_here = 0
        for ln in lines:
            parts = ln.split()
            if len(parts) < 5:
                continue
            try:
                cid = int(float(parts[0]))
            except Exception:
                continue
            total_lines += 1
            if cid in remove_ids:
                removed_here += 1
                continue
            kept.append(parts)
            remaining_ids_set.add(cid)
        files_info[t] = (lines, kept, removed_here)
        total_removed_lines += removed_here

    remaining_ids = sorted(remaining_ids_set)
    print("Total label lines:", total_lines)
    print("Total lines to remove:", total_removed_lines)
    print("Remaining class ids found (before remapping):", remaining_ids)

    # build remapping from old id -> new id (contiguous 0..)
    remap = {}
    for new_i, old in enumerate(remaining_ids):
        remap[old] = new_i
    print("Remap table (old->new):", remap)

    if preview_only:
        # show a small preview of file changes
        sample_changes = 0
        print("\nPreview mode: the following label files (relative to labels dir) would be changed (removed lines count shown):")
        for t, (_, _, removed_here) in list(files_info.items())[:200]:
            if removed_here > 0:
                print(f"  {t}: remove {removed_here} lines")
                sample_changes += 1
        if sample_changes == 0:
            print("  (no files would change in labels dir; ensure LABELS_DIR is correct and files are .txt)")
        print("\nNo files were modified because preview_only=True. To apply changes, set preview_only=False.")
        return

    # do backup if requested
    if do_backup:
        print("Backing up labels dir...")
        b = backup_dir(labels_dir)
        print("Backup created:", b)

    # pass 2: write filtered + remapped labels
    for t, (orig_lines, kept, removed_here) in files_info.items():
        out_lines = []
        for parts in kept:
            old_c = int(float(parts[0]))
            new_c = remap[old_c]
            rest = parts[1:]
            out_lines.append(" ".join([str(new_c)] + rest))
        label_path = os.path.join(labels_dir, t)
        if len(out_lines) == 0:
            # empty result
            open(label_path, 'w', encoding='utf-8').close()  # write empty file
            # optionally move image to removed_images_dir
            if remove_empty_images and images_dir:
                imgname = os.path.splitext(t)[0]
                # find common image extensions
                for ext in ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'):
                    imgpath = os.path.join(images_dir, imgname + ext)
                    if os.path.exists(imgpath):
                        shutil.move(imgpath, os.path.join(removed_images_dir, os.path.basename(imgpath)))
                        print("Moved image with no labels:", imgpath)
                        break
        else:
            with open(label_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(out_lines))

    # update data.yaml: remove the name(s) and set new nc if requested
    if data is not None and update_data_yaml:
        old_names = data.get('names')
        # normalize names to list when possible, but be defensive if format is unexpected
        normalized_names = None
        if isinstance(old_names, dict):
            try:
                idx_names = [None] * (max(int(k) for k in old_names.keys()) + 1)
                for k, v in old_names.items():
                    idx_names[int(k)] = v
                normalized_names = idx_names
            except Exception:
                normalized_names = None
        elif isinstance(old_names, list):
            normalized_names = list(old_names)

        # Build new names list for remaining_ids, using fallbacks when entries are missing
        new_names = []
        for old_id in remaining_ids:
            name = None
            if normalized_names is not None and 0 <= old_id < len(normalized_names):
                name = normalized_names[old_id]
            # try fallback to dict-style mapping from the original data if present
            if name is None:
                raw_names = data.get('names')
                if isinstance(raw_names, dict):
                    name = raw_names.get(str(old_id)) or raw_names.get(int(old_id))
            # final fallback: placeholder name
            if name is None:
                name = f"class{old_id}"
                print(f"Warning: missing name for old id {old_id} in data.yaml; using placeholder '{name}'")
            new_names.append(name)

        data['nc'] = len(new_names)
        # put names as list (Ultralytics accepts list or dict)
        data['names'] = new_names
        print("Updating data.yaml -> nc:", data['nc'], "names:", data['names'])
        if data_yaml_path:
            # backup original data.yaml
            shutil.copyfile(data_yaml_path, data_yaml_path + ".bak")
            save_data_yaml(data_yaml_path, data)
            print("Wrote updated data.yaml (backup created with .bak)")

    print("Done. Labels updated. You have a backup of label files if needed.")
    print("Recommendation: run a quick validation by training for 1 epoch or calling model.val() to ensure everything works.")


if __name__ == '__main__':
    # ====== CONFIGURE THESE VARIABLES BEFORE RUNNING ======
    LABELS_DIR = os.getenv("LABELS_DIR")  # path to your labels folder
    IMAGES_DIR = os.getenv("IMAGES_DIR")  # optional, used only if remove_empty_images=True
    DATA_YAML = os.getenv("YAML_PATH")  # optional path to data.yaml

    # remove either by name (reads names from DATA_YAML) or by id
    REMOVE_BY = 'id'  # 'name' or 'id'
    REMOVE_CLASS_NAME = ['unripe', 'ripe']  # used if REMOVE_BY == 'name'
    REMOVE_CLASS_ID = 1  # used if REMOVE_BY == 'id'

    # Apply behavior
    REMOVE_EMPTY_IMAGES = False  # move images with no labels to labels/removed_images
    DO_BACKUP = True
    UPDATE_DATA_YAML = True

    # If True the script will only print a preview and not modify files. Set to False to apply changes.
    PREVIEW_ONLY = False
    # ======================================================

    run(LABELS_DIR,
        images_dir=IMAGES_DIR,
        data_yaml_path=DATA_YAML,
        remove_by=REMOVE_BY,
        remove_class_name=REMOVE_CLASS_NAME,
        remove_class_id=REMOVE_CLASS_ID,
        remove_empty_images=REMOVE_EMPTY_IMAGES,
        do_backup=DO_BACKUP,
        update_data_yaml=UPDATE_DATA_YAML,
        preview_only=PREVIEW_ONLY)
