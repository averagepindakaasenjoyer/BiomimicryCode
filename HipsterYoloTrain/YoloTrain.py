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

# Augmentation configuration
AUGMENTATION = {
    "enable": True,
    "mosaic": 1.0,              # Critical for small object detection (flowers)
    "mixup": 0.1,               # Regularization via mixing images
    "translate": 0.15,
    "scale": 0.6,               # INCREASED: Training at ~5cm, deployment at ~40cm (8x distance)
                                 # Scale 0.6 = 40-160% size range, simulating 5-40cm+ distances
                                 # Helps model learn flowers at various scales/distances
    "shear": 5.0,
    "degrees": 15.0,
    "perspective": 0.001,       # Increased from 0.0002 for better robustness
    "erasing": 0.02,
    "hsv_h": 0.015,             # Color jittering for flower color invariance
    "hsv_s": 0.7,               # Saturation variation
    "hsv_v": 0.4,               # Brightness variation
}

# Training hyperparameters (weight decay and dropout)
TRAINING_HYP = {
    "weight_decay": 0.001,   # Increased L2 regularization for better generalization
    "dropout": 0.2,          # Enable dropout to prevent overfitting (0.2 is moderate)
}

# Disable chunked/subprocess training - use simple sequential training instead
DISABLE_CHUNKED_TRAINING = False


def apply_dropout(yolo_model, p):
    """Best-effort: set dropout probability `p` on any existing Dropout modules.
    This does not add new Dropout layers; it only updates existing nn.Dropout/Dropout2d/3d modules.
    """
    if p is None or p <= 0.0:
        return 0
    changed = 0
    try:
        import torch.nn as nn
        m = getattr(yolo_model, 'model', None)
        if m is None:
            print("apply_dropout: could not access underlying model (no .model attribute)")
            return 0
        for mm in m.modules():
            if isinstance(mm, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                try:
                    mm.p = float(p)
                    changed += 1
                except Exception:
                    pass
    except Exception as e:
        print("apply_dropout: error while setting dropout:", e)
    if changed == 0:
        print("apply_dropout: no Dropout modules found; consider adding Dropout layers manually if needed")
    else:
        print(f"apply_dropout: updated dropout p={p} on {changed} modules")
    return changed


# -------------------- Hyperparameter tuning config + helper --------------------
# Set `HYPERPARAM_TUNING['enable'] = True` to run automatic tuning before training.
HYPERPARAM_TUNING = {
    'enable': True,
    'model': 'yolo11n.pt',
    'data': 'data.yaml',
    'epochs': 15,             # Reduced from 30 (shorter trials for efficiency)
    'iterations': 150,        # Reasonable number of trials
    'optimizer': 'AdamW',
    'space': {
        'lr0': (1e-4, 1e-2),                    # Narrower, more practical learning rate range
        'lrf': (0.01, 0.1),                    # Final LR ratio (learning rate decay)
        'momentum': (0.6, 0.98),               # Optimizer momentum
        'weight_decay': (0.0, 0.001),          # L2 regularization range
        'hsv_h': (0.0, 0.1),                   # HSV hue shift for flower color invariance
        'hsv_s': (0.0, 0.9),                   # HSV saturation range
        'hsv_v': (0.0, 0.3),                   # HSV value/brightness range
        'degrees': (0.0, 30.0),                # Conservative rotation for flower detection
        'translate': (0.0, 0.3),               # Translation augmentation range
        'scale': (0.0, 0.3),                   # Scale range (preserve small objects)
        'mosaic': (0.0, 1.0),                  # Mosaic augmentation (crucial for small objects!)
        'mixup': (0.0, 0.3),                   # Mixup regularization
    },
    'plots': True,            # Visualize tuning progress
    'save': True,             # Save best tuned model
    'val': True,              # Validate during tuning
}


def run_hyperparameter_tuning(device='cpu'):
    """Run Ultralytics' `model.tune(...)` to search for good hyperparameters.

    This is a convenience wrapper that reads `HYPERPARAM_TUNING` and calls
    `YOLO(...).tune(...)`. It prints progress and handles a compatibility
    fallback if `device` is not an accepted argument to `tune()`.
    """
    if not HYPERPARAM_TUNING.get('enable'):
        print('Hyperparameter tuning disabled (set HYPERPARAM_TUNING["enable"]=True to enable)')
        return None

    model_path = HYPERPARAM_TUNING.get('model')
    print(f"Starting hyperparameter tuning using model={model_path} data={HYPERPARAM_TUNING.get('data')}")
    try:
        m = YOLO(model_path)
    except Exception as e:
        print('Error loading model for tuning:', e)
        return None

    tune_kwargs = {
        'data': HYPERPARAM_TUNING.get('data'),
        'epochs': HYPERPARAM_TUNING.get('epochs', 30),
        'iterations': HYPERPARAM_TUNING.get('iterations', 300),
        'optimizer': HYPERPARAM_TUNING.get('optimizer', 'AdamW'),
        'space': HYPERPARAM_TUNING.get('space'),
        'plots': HYPERPARAM_TUNING.get('plots', False),
        'save': HYPERPARAM_TUNING.get('save', False),
        'val': HYPERPARAM_TUNING.get('val', False),
    }

    # Try passing device if supported; otherwise retry without it.
    try:
        tune_kwargs['device'] = device
        print('Tuning kwargs:', tune_kwargs)
        m.tune(**tune_kwargs)
    except TypeError:
        # older/newer ultralytics may not accept device in tune(); retry without it
        print('model.tune() did not accept `device` kwarg; retrying without device')
        tune_kwargs.pop('device', None)
        m.tune(**tune_kwargs)
    except Exception as e:
        print('Hyperparameter tuning failed:', e)
        return None

    print('Hyperparameter tuning finished. Check generated artifacts (if any).')
    return True


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
    model = YOLO('yolo11n.pt')
    print("Starting training...")

    # GPU-optimized settings for NVIDIA RTX 6000 Ada (48GB VRAM, 32 cores on HIPSTER)
    # For smaller GPUs or memory issues, reduce batch_size to 16 or 8
    batch_size = 32
    workers = 8

    # two-stage training params
    freeze_epochs = 10         # Increased from 2 for better backbone stabilization
    total_epochs = 1000         # Reduced from 1000 (flowers are simple; early stopping will kick in)
    early_stop_patience = 7    # Increased from 3 (was too aggressive: ~35 epochs before stopping)
    chunk_size = 10            # Increased from 5 for more stable training

    if device.startswith("cuda"):
        try:
            print(torch.cuda.memory_summary(device=None, abbreviated=True))
        except Exception:
            pass

    train_common = {
        'data': 'data.yaml',
        'imgsz': 640,
        'batch': batch_size,
        'workers': workers,
        'device': device,
        'weight_decay': TRAINING_HYP.get('weight_decay', 0.0),
        'warmup_epochs': 5.0,    # Gradual learning rate ramp-up
        'warmup_bias_lr': 0.1,   # Warmup for biases
        'label_smoothing': 0.1,  # Regularization via label smoothing
        'patience': 7,           # Patience for early stopping
    }

    # Add augmentation flags if enabled in AUGMENTATION config
    if AUGMENTATION.get('enable'):
        train_common['augment'] = True
        if 'mosaic' in AUGMENTATION:
            train_common['mosaic'] = AUGMENTATION['mosaic']
        if 'mixup' in AUGMENTATION:
            train_common['mixup'] = AUGMENTATION['mixup']
        if 'translate' in AUGMENTATION:
            train_common['translate'] = AUGMENTATION['translate']
        if 'scale' in AUGMENTATION:
            train_common['scale'] = AUGMENTATION['scale']
        if 'shear' in AUGMENTATION:
            train_common['shear'] = AUGMENTATION['shear']
        if 'degrees' in AUGMENTATION:
            train_common['degrees'] = AUGMENTATION['degrees']
        if 'perspective' in AUGMENTATION:
            train_common['perspective'] = AUGMENTATION['perspective']
        if 'erasing' in AUGMENTATION:
            train_common['erasing'] = AUGMENTATION['erasing']
        if 'hsv_h' in AUGMENTATION:
            train_common['hsv_h'] = AUGMENTATION['hsv_h']
        if 'hsv_s' in AUGMENTATION:
            train_common['hsv_s'] = AUGMENTATION['hsv_s']
        if 'hsv_v' in AUGMENTATION:
            train_common['hsv_v'] = AUGMENTATION['hsv_v']


    try:
        # First stage: freeze backbone then train for a few epochs
        print(f"Freezing backbone for {freeze_epochs} epochs (if supported) and training...")
        freeze_backbone(model)
        # use train_common for consistent args and optional augmentation
        model.train(epochs=freeze_epochs, **train_common)

        # Second stage: create a fresh model from the checkpoint produced by the frozen stage,
        # then unfreeze and continue training with early stopping checks. Recreating avoids
        # internal state/overrides issues in Ultralytics when calling train() multiple times on
        # the same YOLO instance.
        print("Preparing to continue training from checkpoint saved after freeze stage...")
        run_dir_after_freeze = latest_train_run_dir()
        freeze_stage_dir = run_dir_after_freeze  # Save for chunk result consolidation
        if freeze_stage_dir is None:
            print("[WARNING] Could not find freeze stage directory - chunk consolidation may fail")
        ckpt = None
        if run_dir_after_freeze is not None:
            # look for last.pt or best.pt
            cand_last = os.path.join(run_dir_after_freeze, 'weights', 'last.pt')
            cand_best = os.path.join(run_dir_after_freeze, 'weights', 'best.pt')
            if os.path.exists(cand_last):
                ckpt = cand_last
            elif os.path.exists(cand_best):
                ckpt = cand_best
        
        if DISABLE_CHUNKED_TRAINING:
            # Simple sequential training without chunking/subprocess
            if ckpt is not None:
                print(f"Loading checkpoint for continued training: {ckpt}")
                model = YOLO(ckpt)
                apply_dropout(model, TRAINING_HYP.get('dropout'))
            else:
                apply_dropout(model, TRAINING_HYP.get('dropout'))
                unfreeze_backbone(model)
            
            remaining_epochs = total_epochs - freeze_epochs
            print(f"Training remaining {remaining_epochs} epochs (unfrozen)...")
            model.train(epochs=remaining_epochs, **train_common)
        
        # Original chunked training logic now disabled - uncomment DISABLE_CHUNKED_TRAINING = False to re-enable
        else:
            # Chunked training: prepare model before starting chunks
            if ckpt is not None:
                print(f"Loading checkpoint for chunked training: {ckpt}")
                model = YOLO(ckpt)
                apply_dropout(model, TRAINING_HYP.get('dropout'))
            else:
                apply_dropout(model, TRAINING_HYP.get('dropout'))
                unfreeze_backbone(model)
            
            remaining = total_epochs - freeze_epochs
            best_map = -1.0
            no_improve = 0
            chunk_num = 0
            cumulative_epochs = freeze_epochs  # Track total epochs completed so far
            for start in range(0, remaining, chunk_size):
                epochs_to_run = min(chunk_size, remaining - start)
                chunk_num += 1
                cumulative_epochs += epochs_to_run  # Update target epoch count
                print(f"Training next chunk of {epochs_to_run} epochs (progress {start + freeze_epochs}/{total_epochs}, target epoch: {cumulative_epochs})...")
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
                        aug_args = ''
                        if AUGMENTATION.get('enable'):
                            aug_args += f", augment={repr(True)}"
                            if 'mosaic' in AUGMENTATION:
                                aug_args += f", mosaic={repr(AUGMENTATION['mosaic'])}"
                            if 'mixup' in AUGMENTATION:
                                aug_args += f", mixup={repr(AUGMENTATION['mixup'])}"
                            if 'translate' in AUGMENTATION:
                                aug_args += f", translate={repr(AUGMENTATION['translate'])}"
                            if 'scale' in AUGMENTATION:
                                aug_args += f", scale={repr(AUGMENTATION['scale'])}"
                            if 'shear' in AUGMENTATION:
                                aug_args += f", shear={repr(AUGMENTATION['shear'])}"
                            if 'degrees' in AUGMENTATION:
                                aug_args += f", degrees={repr(AUGMENTATION['degrees'])}"
                            if 'perspective' in AUGMENTATION:
                                aug_args += f", perspective={repr(AUGMENTATION['perspective'])}"
                            if 'erasing' in AUGMENTATION:
                                aug_args += f", erasing={repr(AUGMENTATION['erasing'])}"
                            if 'hsv_h' in AUGMENTATION:
                                aug_args += f", hsv_h={repr(AUGMENTATION['hsv_h'])}"
                            if 'hsv_s' in AUGMENTATION:
                                aug_args += f", hsv_s={repr(AUGMENTATION['hsv_s'])}"
                            if 'hsv_v' in AUGMENTATION:
                                aug_args += f", hsv_v={repr(AUGMENTATION['hsv_v'])}"

                        # add weight_decay to subprocess args
                        # Note: NO warmup for chunks - we want LR to continue from previous chunk
                        # Only the freeze stage uses warmup_epochs=5.0 above
                        aug_args += f", weight_decay={repr(TRAINING_HYP.get('weight_decay', 0.0))}"
                        aug_args += f", warmup_epochs=0, label_smoothing=0.1, patience=10"
                        
                        # Read the last learning rate from previous chunk to continue the schedule
                        # This is much better than calculating it - uses the actual LR that was used
                        last_lr = parse_lr_from_results_csv(rerun_ckpt_dir)
                        if last_lr is not None:
                            current_lr0 = last_lr
                            print(f"Continuing from previous LR: {current_lr0:.6f}")
                        else:
                            # Fallback: start with a reasonable default for later chunks
                            current_lr0 = 0.001
                            print(f"Could not read previous LR, using fallback: {current_lr0:.6f}")
                        aug_args += f", lr0={current_lr0}, lrf=0.01"

                        # Simplified subprocess training - remove dropout modification as it causes issues
                        # Dropout should already be in the checkpoint from the initial training phase
                        # Don't use resume=True - we want separate directories per chunk for tracking
                        # Instead, continue LR schedule by reading from previous chunk's CSV
                        code = (
                            f"from ultralytics import YOLO;"
                            f"YOLO({repr(rerun_ckpt)}).train(data='data.yaml', epochs={epochs_to_run}, "
                            f"imgsz=640, device={repr(device)}, batch={batch_size}, workers={workers}{aug_args})"
                        )
                        
                        # Track the training directory BEFORE subprocess
                        run_dir_before = latest_train_run_dir()
                        
                        try:
                            # Don't capture output so we can see real-time progress and actual errors
                            result = subprocess.run([sys.executable, '-c', code], check=True)
                            
                            # Verify a new train directory was created
                            run_dir_after = latest_train_run_dir()
                            if run_dir_after == run_dir_before:
                                print(f"[WARNING] Chunk {chunk_num}: Subprocess completed but no new training directory created!")
                                print(f"  Before: {run_dir_before}")
                                print(f"  After: {run_dir_after}")
                                print(f"  This suggests training may not have actually occurred.")
                            
                            print(f"✓ Chunk {chunk_num} training completed successfully")
                        except subprocess.CalledProcessError as e:
                            print(f"✗ Subprocess training chunk {chunk_num} failed with exit code {e.returncode}")
                            print(f"Command: {code}")
                            print("Stopping chunked training due to subprocess failure")
                            break  # Stop training but don't crash - we have partial results
                    else:
                        print(f"[ERROR] No checkpoint found for subprocess chunk {chunk_num}")
                        print(f"  Candidates checked: {rerun_ckpt_dir}")
                        print(f"  Expected: {os.path.join(rerun_ckpt_dir, 'weights', 'last.pt') if rerun_ckpt_dir else 'N/A'}")
                        print("Stopping chunked training due to missing checkpoint")
                        break
                else:
                    # No checkpoint from latest run - reload from initial checkpoint or use prepared model
                    # This fallback should rarely happen (only on first chunk if subprocess fails)
                    print("No checkpoint found for subprocess, using in-process training for this chunk")
                    latest_ckpt = None
                    run_dir = latest_train_run_dir()
                    if run_dir:
                        cand = os.path.join(run_dir, 'weights', 'last.pt')
                        if os.path.exists(cand):
                            latest_ckpt = cand
                    
                    if latest_ckpt:
                        print(f"Loading {latest_ckpt} for in-process training")
                        chunk_model = YOLO(latest_ckpt)
                    elif ckpt:
                        print(f"Loading initial checkpoint {ckpt} for in-process training")
                        chunk_model = YOLO(ckpt)
                    else:
                        print("ERROR: No checkpoint available for chunk training - cannot continue")
                        break
                    
                    # Train in-process with cumulative epoch target
                    try:
                        # Read LR from previous chunk's results
                        last_lr = parse_lr_from_results_csv(run_dir) if run_dir else None
                        if last_lr is not None:
                            current_lr0 = last_lr
                            print(f"Continuing from previous LR: {current_lr0:.6f}")
                        else:
                            current_lr0 = 0.001
                            print(f"Could not read previous LR, using fallback: {current_lr0:.6f}")
                        
                        print(f"Training in-process (target: {epochs_to_run} epochs, lr0={current_lr0:.6f})...")
                        # Use a copy of train_common but override warmup for chunks
                        chunk_train_params = {**train_common, 'lr0': current_lr0, 'lrf': 0.01, 'warmup_epochs': 0}
                        chunk_model.train(epochs=epochs_to_run, **chunk_train_params)
                        print(f"✓ Chunk {chunk_num} in-process training completed")
                    except Exception as e:
                        print(f"✗ In-process chunk training failed: {e}")
                        break

                # after chunk, parse latest results.csv and check mAP50
                run_dir = latest_train_run_dir()
                if run_dir is None:
                    print("Warning: could not find training run results to evaluate early stopping.")
                else:
                    map50, map5095 = parse_map_from_results_csv(run_dir)
                    if map50 is not None:
                        print(f"Chunk results: mAP50={map50:.4f}, mAP50-95={map5095:.4f}")
                        
                        # Copy results.csv to freeze_stage_dir with chunk number for consolidation
                        src_csv = os.path.join(run_dir, 'results.csv')
                        if freeze_stage_dir and os.path.exists(src_csv):
                            try:
                                dst_csv = os.path.join(freeze_stage_dir, f'results_chunk_{chunk_num:02d}.csv')
                                import shutil
                                shutil.copy(src_csv, dst_csv)
                                print(f"Saved chunk results to {dst_csv}")
                            except Exception as e:
                                print(f"[WARNING] Failed to copy chunk results: {e}")
                        elif not freeze_stage_dir:
                            print("[WARNING] freeze_stage_dir not set - chunk results not saved for consolidation")
                        
                        # Early stopping logic: check if mAP improved
                        if map50 > best_map:
                            best_map = map50
                            no_improve = 0
                            print(f"New best mAP50: {best_map:.4f} (improvement from previous best)")
                        else:
                            no_improve += 1
                            print(f"No improvement count: {no_improve}/{early_stop_patience} (best mAP50 so far: {best_map:.4f})")
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
    
    # Debug: show current working directory
    cwd = os.getcwd()
    print(f"[DEBUG] Current working directory: {cwd}")
    
    # Look specifically for training runs, not validation runs
    search_patterns = [
        os.path.join('runs', 'detect', 'train*'),  # runs/detect/train, train2, train3, etc.
        os.path.join('runs', 'train', '*'),        # runs/train/*
    ]
    candidates = []
    for pat in search_patterns:
        matches = glob.glob(pat)
        print(f"[DEBUG] Pattern '{pat}' matched {len(matches)} directories")
        for p in matches:
            if os.path.isdir(p):
                # Exclude validation directories
                basename = os.path.basename(p)
                if 'val' not in basename.lower() and 'predict' not in basename.lower():
                    candidates.append(p)
                    print(f"[DEBUG]   Added candidate: {p}")

    # deduplicate
    candidates = sorted(list(dict.fromkeys(candidates)))
    print(f"[DEBUG] Found {len(candidates)} candidate directories (after filtering)")
    if not candidates:
        print("[DEBUG] No training run directories found")
        return None
    
    # Sort by directory number (train -> 0, train2 -> 2, train37 -> 37, etc.)
    # This ensures we get the highest-numbered (most recent) training run
    def extract_train_number(path):
        basename = os.path.basename(path)
        # Extract number from 'train', 'train2', 'train37', etc.
        if basename == 'train':
            return 0
        elif basename.startswith('train'):
            try:
                return int(basename[5:])  # 'train37' -> 37
            except ValueError:
                return -1
        return -1
    
    candidates.sort(key=extract_train_number, reverse=True)
    latest = candidates[0]
    print(f"[DEBUG] Latest training run directory (by number): {latest}")
    
    # Verify it has weights folder
    weights_dir = os.path.join(latest, 'weights')
    if os.path.exists(weights_dir):
        print(f"[DEBUG] Confirmed weights directory exists: {weights_dir}")
    else:
        print(f"[DEBUG] WARNING: No weights directory in {latest}")
    
    return latest


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


def parse_lr_from_results_csv(run_dir):
    """Read the last learning rate from results.csv.
    Returns the lr/pg0 value (primary learning rate) from the last epoch, or None if not found."""
    csv_path = os.path.join(run_dir, 'results.csv')
    if not os.path.exists(csv_path):
        return None
    try:
        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)
            if len(rows) < 2:
                return None
            header = rows[0]
            last = rows[-1]
            # Look for lr/pg0, lr/pg1, or similar learning rate columns
            lr_idx = None
            for i, col in enumerate(header):
                col_clean = col.strip().lower()
                if 'lr/pg0' in col_clean or col_clean == 'lr' or 'learning_rate' in col_clean:
                    lr_idx = i
                    break
            if lr_idx is not None:
                lr_val = float(last[lr_idx])
                return lr_val
            return None
    except Exception as e:
        print(f"Error parsing learning rate from results.csv: {e}")
        return None


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



