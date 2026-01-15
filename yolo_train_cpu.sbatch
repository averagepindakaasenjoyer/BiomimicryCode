#!/bin/bash
#SBATCH -J yolo-cpu-train
#SBATCH -p capacity
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH -t 48:00:00
#SBATCH -o /home/%u/logs/%x-%j.out
#SBATCH -e /home/%u/logs/%x-%j.err

# HIPSTER CPU-only Training Job Script
# Submit: sbatch yolo_train_cpu.sbatch
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

# Stage project to local scratch
echo "[1/4] Staging project to local scratch..."
mkdir -p "$TMPDIR/BiomimicryCode"
rsync -av --delete /home/15178420/BiomimicryCode/ "$TMPDIR/BiomimicryCode/" \
  --exclude "runs" --exclude "training_logs" --exclude ".git" \
  || true

cd "$TMPDIR/BiomimicryCode"
echo "[2/4] Working directory: $(pwd)"

# Verify data.yaml exists
if [ ! -f data.yaml ]; then
  echo "ERROR: data.yaml not found in $(pwd)"
  exit 1
fi

# Run training on CPU (set workers to match available cores)
echo "[3/4] Starting YOLO training on CPU (32 cores)..."
# Note: YoloTrain.py hardcodes workers=2; for CPU-only, you may want more workers.
# If needed, modify YoloTrain.py to accept --workers or set via env var.
python server_train.py --plots --report

# Copy results back to home
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
