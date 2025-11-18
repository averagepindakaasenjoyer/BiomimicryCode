import cv2
import numpy as np
from ultralytics import YOLO
import os
import dotenv

import torch
import time
import glob
import csv
import matplotlib.pyplot as plt
import subprocess
import sys

dotenv.load_dotenv()


def check_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
        try:
            dev_name = torch.cuda.get_device_name(0)
        except Exception:
            dev_name = 'GPU'
        print(f"CUDA is available. Using GPU: {dev_name} (device={device})")
    else:
        device = 'cpu'
        print("CUDA not available — falling back to CPU. Training will be much slower.")
    return device


def run_training_validation(device):
    model = YOLO('yolov8n.pt')
    print("Starting training...")

    # Recommended safe defaults to avoid DataLoader / memory issues on small machines:
    batch_size = 8
    workers = 2

    # two-stage training params
    freeze_epochs = 10  # freeze backbone for these first epochs
    total_epochs = 1000
    early_stop_patience = 3  # number of chunks with no improvement before stopping
    chunk_size = 5  # train in chunks after unfreezing to allow early stopping checks

    if device.startswith("cuda"):
        try:
            print(torch.cuda.memory_summary(device=None, abbreviated=True))
        except Exception:
            pass

    try:
        # First stage: freeze backbone then train for a few epochs
        print(f"Freezing backbone for {freeze_epochs} epochs (if supported) and training...")
        freeze_backbone(model)
        model.train(data='data.yaml', epochs=freeze_epochs, imgsz=640, device=device, batch=batch_size, workers=workers)

        # Second stage: create a fresh model from the checkpoint produced by the frozen stage,
        # then unfreeze and continue training with early stopping checks. Recreating avoids
        # internal state/overrides issues in Ultralytics when calling train() multiple times on
        # the same YOLO instance.
        print("Preparing to continue training from checkpoint saved after freeze stage...")
        run_dir_after_freeze = latest_train_run_dir()
        ckpt = None
        if run_dir_after_freeze is not None:
            # look for last.pt or best.pt
            cand_last = os.path.join(run_dir_after_freeze, 'weights', 'last.pt')
            cand_best = os.path.join(run_dir_after_freeze, 'weights', 'best.pt')
            if os.path.exists(cand_last):
                ckpt = cand_last
            elif os.path.exists(cand_best):
                ckpt = cand_best
        if ckpt is None:
            print("Warning: could not find checkpoint from freeze stage; continuing with original model instance.")
            unfreeze_backbone(model)
            working_model = model
        else:
            print(f"Loading checkpoint for continued training: {ckpt}")
            working_model = YOLO(ckpt)
            unfreeze_backbone(working_model)

        remaining = total_epochs - freeze_epochs
        best_map = -1.0
        no_improve = 0
        for start in range(0, remaining, chunk_size):
            epochs_to_run = min(chunk_size, remaining - start)
            print(f"Training next chunk of {epochs_to_run} epochs (progress {start + freeze_epochs}/{total_epochs})...")
            # Run chunked training in a fresh Python process using the latest checkpoint (if available).
            # This avoids Ultralytics internal state issues when calling train() repeatedly in the same process.
            rerun_ckpt_dir = latest_train_run_dir()
            rerun_ckpt = None
            if rerun_ckpt_dir:
                candidate = os.path.join(rerun_ckpt_dir, 'weights', 'last.pt')
                if os.path.exists(candidate):
                    rerun_ckpt = candidate
                else:
                    candidate_best = os.path.join(rerun_ckpt_dir, 'weights', 'best.pt')
                    if os.path.exists(candidate_best):
                        rerun_ckpt = candidate_best

            if rerun_ckpt is not None:
                print(f"Launching subprocess training chunk from checkpoint: {rerun_ckpt}")
                code = (
                    f"from ultralytics import YOLO;"
                    f"YOLO({repr(rerun_ckpt)}).train(data='data.yaml', epochs={epochs_to_run}, imgsz=640, device={repr(device)}, batch={batch_size}, workers={workers})"
                )
                try:
                    subprocess.run([sys.executable, '-c', code], check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Subprocess training chunk failed: {e}")
                    raise
            else:
                # Fall back to in-process training if no checkpoint is available
                working_model.train(data='data.yaml', epochs=epochs_to_run, imgsz=640, device=device, batch=batch_size, workers=workers)

            # after chunk, parse latest results.csv and check mAP50
            run_dir = latest_train_run_dir()
            if run_dir is None:
                print("Warning: could not find training run results to evaluate early stopping.")
            else:
                map50, map5095 = parse_map_from_results_csv(run_dir)
                if map50 is not None:
                    print(f"Chunk results: mAP50={map50:.4f}, mAP50-95={map5095:.4f}")
                    if map50 > best_map:
                        best_map = map50
                        no_improve = 0
                    else:
                        no_improve += 1
                        print(f"No improvement count: {no_improve}/{early_stop_patience}")
                else:
                    print("Could not read mAP from results.csv for this chunk.")

            if no_improve >= early_stop_patience:
                print("Early stopping: no improvement observed for several chunks. Stopping training.")
                break

    except RuntimeError as e:
        msg = str(e)
        print("Training failed with error:\n", msg)
        if "MemoryError" in msg or "out of memory" in msg.lower():
            print("Detected memory error. Suggestions:\n"
                  " - Reduce batch size (e.g. batch=2 or 1)\n"
                  " - Set workers=0 to disable multiprocessing data loaders\n"
                  " - Ensure other processes aren't consuming RAM/GPU memory\n"
                  " - If using CPU only, reduce imgsz or batch size significantly")
        raise

    print("Training completed.")

    # final validation and plotting
    model.val(device=device)
    print("Validation completed.")
    run_dir = latest_train_run_dir()
    if run_dir:
        plot_map_progress(run_dir)


def freeze_backbone(yolo_model):
    """Try multiple strategies to freeze the backbone parameters.
    This function is best-effort: it attempts to call a builtin freeze method,
    or falls back to freezing parameters whose name contains 'backbone'."""
    try:
        if hasattr(yolo_model, 'freeze'):
            yolo_model.freeze()
            print("Called model.freeze()")
            return
    except Exception:
        pass

    # Fallback: try to find parameters belonging to backbone by name heuristics
    m = getattr(yolo_model, 'model', None)
    if m is None:
        print("Warning: could not access underlying model to freeze backbone.")
        return

    frozen = 0
    for name, p in m.named_parameters():
        if 'backbone' in name or 'backbone' in name.lower() or name.startswith('model.0'):
            p.requires_grad = False
            frozen += 1
    print(f"Frozen {frozen} parameters by heuristic.")


def unfreeze_backbone(yolo_model):
    try:
        if hasattr(yolo_model, 'unfreeze'):
            yolo_model.unfreeze()
            print("Called model.unfreeze()")
            return
    except Exception:
        pass

    m = getattr(yolo_model, 'model', None)
    if m is None:
        print("Warning: could not access underlying model to unfreeze backbone.")
        return

    unfrozen = 0
    for name, p in m.named_parameters():
        if not p.requires_grad:
            p.requires_grad = True
            unfrozen += 1
    print(f"Unfroze {unfrozen} parameters by heuristic.")


def latest_train_run_dir():
    """Find the most recent training run directory produced by Ultralyics.
    Searches common locations: runs/train/*, runs/detect/*, and runs/*/train*.
    Returns the most recently modified directory path or None if none found."""
    search_patterns = [
        os.path.join('runs', 'train', '*'),
        os.path.join('runs', 'detect', '*'),
        os.path.join('runs', '*', 'train*'),
        os.path.join('runs', '*', 'exp*'),
        os.path.join('runs', '*', '*train*'),
    ]
    candidates = []
    for pat in search_patterns:
        for p in glob.glob(pat):
            if os.path.isdir(p):
                candidates.append(p)

    # also consider any directory directly under runs that looks like a training run
    for p in glob.glob(os.path.join('runs', '*')):
        if os.path.isdir(p) and ('train' in os.path.basename(p).lower() or 'exp' in os.path.basename(p).lower() or 'detect' in os.path.basename(p).lower()):
            candidates.append(p)

    # deduplicate
    candidates = sorted(list(dict.fromkeys(candidates)))
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def parse_map_from_results_csv(run_dir):
    # results.csv typically in run_dir/results.csv with header that contains mAP columns
    csv_path = os.path.join(run_dir, 'results.csv')
    if not os.path.exists(csv_path):
        return None, None
    try:
        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)
            if len(rows) < 2:
                return None, None
            header = rows[0]
            last = rows[-1]
            # try common header names
            map50_idx = None
            map5095_idx = None
            for i, col in enumerate(header):
                key = col.lower()
                if 'map_0.5' in key or 'map50' in key or 'map50' in col:
                    map50_idx = i
                if 'map_0.5:0.95' in key or 'map50-95' in key or 'map_0.5-0.95' in key or 'map5095' in key:
                    map5095_idx = i
            # fallback heuristics: try last columns
            if map50_idx is None and len(header) >= 3:
                map50_idx = -3
            if map5095_idx is None and len(header) >= 3:
                map5095_idx = -2

            map50 = float(last[map50_idx]) if map50_idx is not None else None
            map5095 = float(last[map5095_idx]) if map5095_idx is not None else None
            return map50, map5095
    except Exception as e:
        print(f"Error parsing results.csv: {e}")
        return None, None


def plot_map_progress(run_dir, save_path=None):
    csv_path = os.path.join(run_dir, 'results.csv')
    if not os.path.exists(csv_path):
        print("No results.csv found to plot.")
        return
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)
        if len(rows) < 2:
            print("Not enough rows in results.csv to plot.")
            return
        header = rows[0]
        data = rows[1:]
        # find indices
        map50_idx = None
        map5095_idx = None
        for i, col in enumerate(header):
            key = col.lower()
            if 'map_0.5' in key or 'map50' in key or 'map50' in col:
                map50_idx = i
            if 'map_0.5:0.95' in key or 'map50-95' in key or 'map_0.5-0.95' in key or 'map5095' in key:
                map5095_idx = i

        map50_vals = []
        map5095_vals = []
        for row in data:
            try:
                if map50_idx is not None:
                    map50_vals.append(float(row[map50_idx]))
                else:
                    map50_vals.append(None)
                if map5095_idx is not None:
                    map5095_vals.append(float(row[map5095_idx]))
                else:
                    map5095_vals.append(None)
            except Exception:
                map50_vals.append(None)
                map5095_vals.append(None)

        epochs = list(range(1, len(map50_vals) + 1))
        plt.figure()
        if any(v is not None for v in map50_vals):
            plt.plot(epochs, map50_vals, label='mAP50')
        if any(v is not None for v in map5095_vals):
            plt.plot(epochs, map5095_vals, label='mAP50-95')
        plt.xlabel('Epoch')
        plt.ylabel('mAP')
        plt.legend()
        plt.grid(True)
        if save_path is None:
            save_path = os.path.join(run_dir, 'map_progress.png')
        plt.savefig(save_path)
        print(f"Saved mAP plot to {save_path}")


def gradcam_visualize(model, image_path, out_path):
    """Try to create a Grad-CAM visualization for a single image using pytorch-grad-cam.
    If the library isn't available, print instructions to install it.
    This is a best-effort helper and may need adaptation for YOLO's model internals."""
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except Exception:
        print("pytorch-grad-cam not installed. Install with:\n    pip install grad-cam")
        return

    # load image
    import cv2
    img = cv2.imread(image_path)[:, :, ::-1]  # BGR->RGB
    img_input = cv2.resize(img, (640, 640)) / 255.0
    input_tensor = torch.from_numpy(img_input).permute(2, 0, 1).unsqueeze(0).float()

    # find a conv layer to target
    backbone = getattr(model, 'model', None)
    target_layer = None
    if backbone is not None:
        for m in reversed(list(backbone.modules())):
            if isinstance(m, torch.nn.Conv2d):
                target_layer = m
                break
    if target_layer is None:
        print("Could not find a conv layer to use for Grad-CAM.")
        return

    cam = GradCAM(model=model.model, target_layers=[target_layer], use_cuda=torch.cuda.is_available())
    targets = [ClassifierOutputTarget(0)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    visualization = show_cam_on_image(img_input, grayscale_cam, use_rgb=True)
    cv2.imwrite(out_path, visualization[:, :, ::-1])
    print(f"Saved Grad-CAM visualization to {out_path}")

if __name__ == '__main__':
    device = check_device()
    run_training_validation(device)



