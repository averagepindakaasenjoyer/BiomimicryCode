#!/bin/bash
#SBATCH -J yolo-gpu-train
#SBATCH -p performance
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH -t 24:00:00
#SBATCH -o /home/%u/logs/%x-%j.out
#SBATCH -e /home/%u/logs/%x-%j.err

# HIPSTER GPU Training Job Script
# Submit: sbatch yolo_train_gpu.sbatch
# Monitor: squeue -u $USER
# Cancel: scancel <job_id>
# ⚠️ Note: /home has 200GiB limit. Monitor with: quota -s

set -e

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "TMPDIR: $TMPDIR"
echo ""

# Setup modules
module use /cvmfs/software.eessi.io/init/modules
module load EESSI/2023.06
module load Python/3.11.5-GCCcore-13.2.0

# Activate venv
source ~/envs/yolo-hipster/bin/activate

# Create log directory
mkdir -p /home/$USER/logs

# Project root in home (as observed on HIPSTER: /home/$USER/Biomimicry/BiomimicryCode)
PROJECT_ROOT="/home/$USER/Biomimicry/BiomimicryCode"

# Stage project to fast local scratch (NVMe)
echo "[1/4] Staging project to local scratch..."
mkdir -p "$TMPDIR/BiomimicryCode"
rsync -av --delete "$PROJECT_ROOT/" "$TMPDIR/BiomimicryCode/" \
  --exclude "runs" --exclude "training_logs" --exclude ".git" \
  --exclude "__pycache__" --exclude "*.pyc" \
  --exclude "Combined_Dataset*" \
  || true  # tolerate minor rsync issues

# Copy dataset to scratch (can be large, allow time)
echo "[1.5/4] Copying dataset to local scratch (this may take a few minutes)..."
BIOMIMICRY_ROOT="$(dirname "$PROJECT_ROOT")"

if [ -d "$BIOMIMICRY_ROOT/Combined_Dataset2" ]; then
  timeout 600 cp -r "$BIOMIMICRY_ROOT/Combined_Dataset2" "$TMPDIR/BiomimicryCode/" || {
    echo "WARNING: Dataset copy timed out or failed. Training may fail if dataset is needed."
  }
elif [ -d "$BIOMIMICRY_ROOT/Combined_Dataset" ]; then
  timeout 600 cp -r "$BIOMIMICRY_ROOT/Combined_Dataset" "$TMPDIR/BiomimicryCode/" || {
    echo "WARNING: Dataset copy timed out or failed. Training may fail if dataset is needed."
  }
else
  echo "WARNING: No Combined_Dataset found in $BIOMIMICRY_ROOT"
  echo "Available directories:"
  ls -la "$BIOMIMICRY_ROOT" | grep -i dataset || echo "  (none found)"
fi

cd "$TMPDIR/BiomimicryCode"
echo "[2/4] Working directory: $(pwd)"

# Verify data.yaml exists
if [ ! -f data.yaml ]; then
  echo "ERROR: data.yaml not found in $(pwd)"
  exit 1
fi

# Run training with GPU (outputs to $TMPDIR/runs/)
echo "[3/4] Starting YOLO training on GPU..."
python server_train.py --plots --report

# Copy results back to home (persistent storage)
echo "[4/4] Copying results back to $PROJECT_ROOT..."
mkdir -p "$PROJECT_ROOT/runs"
mkdir -p "$PROJECT_ROOT/training_logs"
rsync -av runs/ "$PROJECT_ROOT/runs/" || true
rsync -av training_logs/ "$PROJECT_ROOT/training_logs/" || true

# Cleanup: remove dataset from scratch to free space
echo "[5/5] Cleaning up scratch space..."
rm -rf "$TMPDIR/BiomimicryCode/Combined_Dataset2" "$TMPDIR/BiomimicryCode/Combined_Dataset" || true

echo ""
echo "Job completed at $(date)"
echo "Results available at: $PROJECT_ROOT/runs"
echo "Logs available at: $PROJECT_ROOT/training_logs"
echo ""
echo "Check home quota: quota -s"
