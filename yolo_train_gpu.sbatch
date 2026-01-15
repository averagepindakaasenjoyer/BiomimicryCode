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
# Monitor: squeue -u 15178420
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
module load Python-bundle-PyPI/2023.10-GCCcore-13.2.0
module load PyYAML/6.0.1-GCCcore-13.2.0

# Activate venv with ultralytics
source ~/envs/yolo-hipster/bin/activate

# Create log directory
mkdir -p /home/15178420/logs

# Stage project to fast local scratch (NVMe)
echo "[1/4] Staging project to local scratch..."
mkdir -p "$TMPDIR/BiomimicryCode"
rsync -av --delete /home/15178420/BiomimicryCode/ "$TMPDIR/BiomimicryCode/" \
  --exclude "runs" --exclude "training_logs" --exclude ".git" \
  || true  # tolerate minor rsync issues

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
echo "[4/4] Copying results back to /home/15178420..."
mkdir -p /home/15178420/BiomimicryCode/runs
mkdir -p /home/15178420/BiomimicryCode/training_logs
rsync -av runs/ /home/15178420/BiomimicryCode/runs/ || true
rsync -av training_logs/ /home/15178420/BiomimicryCode/training_logs/ || true

echo ""
echo "Job completed at $(date)"
echo "Results available at: /home/15178420/BiomimicryCode/runs"
echo "Logs available at: /home/15178420/BiomimicryCode/training_logs"
echo ""
echo "Check home quota: quota -s"
