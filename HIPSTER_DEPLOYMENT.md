# HIPSTER Cluster Deployment Guide for YOLO Training

## Environment Setup (One-Time on Login Node)

Run this once to install ultralytics and torch in a venv:

```bash
# Copy setup script to login node and run
bash setup_env_hipster.sh

# This:
# 1. Loads EESSI modules (Python 3.11, numpy, matplotlib, pyyaml, etc.)
# 2. Creates ~/envs/yolo-hipster venv
# 3. Installs ultralytics, torch, torchvision, opencv-python, python-dotenv

# After setup, verify:
source ~/envs/yolo-hipster/bin/activate
python -c "import torch, ultralytics, cv2; print('OK')"
```

## Data Preparation

Ensure your dataset is copied to `/home/15178420/BiomimicryCode/`:

```bash
# On login node:
rsync -av ~/BiomimicryCode/ /home/15178420/BiomimicryCode/

# Verify data.yaml paths are accessible:
head data.yaml
```

**⚠️ Storage Policy & Quota:**
- `/home`: **200GiB limit per user** ← *Your code + models + training outputs must fit here*
- `/scratch/15178420`: 2TiB/user, auto-cleanup after 2 weeks (alternative if you exceed home quota)
- `/fnwi_fs/fnwi/hipster`: Long-term bulk data, up to 10TiB/user (request if needed)

**For training in `/home`:**
- Working directory: `/home/15178420/BiomimicryCode`
- Job scripts stage code+data to `$TMPDIR` (local NVMe, fastest) for training
- Results rsync back to `/home/15178420/BiomimicryCode` after training completes
- **Monitor quota**: Run `quota -s` on login node to check usage
- **If quota fills**: Delete old `runs/` directories or move to `/scratch` temporarily

## Running Training

### Option 1: GPU Training (Recommended)

```bash
# Submit GPU job (performance partition, 1 GPU, ~5–12 hours depending on epochs)
sbatch yolo_train_gpu.sbatch

# Monitor:
squeue -u 15178420

# Cancel if needed:
scancel <job_id>

# View output/errors (real-time):
mkdir -p /home/15178420/logs  # Create logs directory if needed
tail -f /home/15178420/logs/yolo-gpu-train-<job_id>.out
tail -f /home/15178420/logs/yolo-gpu-train-<job_id>.err
```

**GPU Resources:**
- Partition: `performance` (8 GPUs/node, 5 nodes total)
- 1 GPU + 32 cores + 96GB RAM (fractional allocation per GPU spec)
- Time limit: 24 hours

### Option 2: CPU-Only Training

```bash
# For CPU-only (capacity partition, slower but useful for debugging):
sbatch yolo_train_cpu.sbatch

# Monitor:
squeue -u 15178420
tail -f /home/15178420/logs/yolo-cpu-train-<job_id>.out
```

**CPU Resources:**
- Partition: `capacity` (8 GPUs/node, 8 nodes, but job runs on CPU only)
- 32 cores + 192GB RAM
- Time limit: 48 hours

### Option 3: Interactive Testing

```bash
# For quick debugging (CPU):
srun -p capacity --cpus-per-task=8 --mem=32G --pty bash

module use /cvmfs/software.eessi.io/init/modules
module load EESSI/2023.06
module load Python/3.11.5-GCCcore-13.2.0
module load Python-bundle-PyPI/2023.10-GCCcore-13.2.0
module load PyYAML/6.0.1-GCCcore-13.2.0

source ~/envs/yolo-hipster/bin/activate
cd /home/15178420/BiomimicryCode
python server_train.py --plots
```

## Monitoring and Retrieving Results

After job completes:

```bash
# Check consolidated results (from login node):
ls -lh /home/15178420/BiomimicryCode/runs/detect/train*/consolidated_results.csv
cat /home/15178420/BiomimicryCode/runs/detect/train*/TRAINING_SUMMARY.txt

# Download to local machine (from your Windows machine, PowerShell or bash):
scp -r <user>@hipster.science.uva.nl:/home/<user>/BiomimicryCode/training_results ./BiomimicryCode_results

# Or use rsync for faster incremental transfers:
rsync -avz --delete <user>@hipster.science.uva.nl:/home/<user>/BiomimicryCode/training_results/ ./BiomimicryCode_results/

# Or use WinSCP/FileZilla for GUI transfer
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'ultralytics'` | Run `setup_env_hipster.sh`, then activate venv: `source ~/envs/yolo-hipster/bin/activate` |
| Job stuck/timeout | Increase `-t` (time limit) in `.sbatch` script; GPU training with 200 epochs may need 24+ hours |
| `data.yaml` not found | Verify file is in working directory; use absolute path or relative path from job cwd |
| Low GPU memory | Reduce `batch_size` in YoloTrain.py (default 8); try 4 or 2 |
| `/home` quota full (200GiB exceeded) | Delete old `runs/` dirs: `rm -rf /home/15178420/BiomimicryCode/runs/detect/train1`; or move to `/scratch` temporarily: `rsync -av /home/15178420/BiomimicryCode/runs /scratch/15178420/` |

## Advanced: Custom Resource Allocation

Adjust job script `#SBATCH` directives to request more/fewer resources:

```bash
# More GPUs (e.g., 2 GPUs):
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=64
#SBATCH --mem=192G

# Shorter wall-time (e.g., 4 hours for quick test):
#SBATCH -t 04:00:00

# Different partition (see HIPSTER docs for fp64 option):
#SBATCH -p fp64
```

Follow the cluster policy: fractional allocation = 32 cores + 96GB per GPU on performance.

## Contact & Policy Reminders

- Report issues: feiog@uva.nl
- Be courteous: don't claim all cluster resources; allow space for colleagues
- Clean up `/scratch` after training; data auto-removes after 2 weeks
- For publications mentioning HIPSTER, acknowledge the facility to justify continued investment

---

**Quick Start Summary:**
1. Login to hipster.science.uva.nl
2. Run `bash setup_env_hipster.sh`
3. Ensure `/scratch/15178420/BiomimicryCode` has your data and model
4. Submit: `sbatch yolo_train_gpu.sbatch` (or cpu variant)
5. Monitor: `squeue -u 15178420` and tail logs
6. Download results after job completes
