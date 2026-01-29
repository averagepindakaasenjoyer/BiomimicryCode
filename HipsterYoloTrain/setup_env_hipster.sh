#!/bin/bash
# Setup environment on HIPSTER login node (one-time)
# Run from login node: bash setup_env_hipster.sh

set -e

echo "=== HIPSTER Environment Setup for YOLO Training ==="
echo ""

# Load EESSI modules
echo "[1/4] Loading EESSI software collection..."
module use /cvmfs/software.eessi.io/init/modules
module load EESSI/2023.06

# Load Python and bundled packages
echo "[2/4] Loading Python 3.11 and dependencies..."
module load Python/3.11.5-GCCcore-13.2.0
module load Python-bundle-PyPI/2023.10-GCCcore-13.2.0
module load PyYAML/6.0.1-GCCcore-13.2.0

# Create venv in home for user-only packages
echo "[3/4] Creating Python virtual environment..."
python -m venv ~/envs/yolo-hipster
source ~/envs/yolo-hipster/bin/activate

# Upgrade pip and install ML packages not in EESSI
echo "[4/4] Installing ultralytics, torch, and opencv-python..."
pip install --upgrade pip
pip install ultralytics torch torchvision opencv-python python-dotenv

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To use this environment in job scripts or interactively:"
echo "  module use /cvmfs/software.eessi.io/init/modules"
echo "  module load EESSI/2023.06"
echo "  module load Python/3.11.5-GCCcore-13.2.0"
echo "  module load Python-bundle-PyPI/2023.10-GCCcore-13.2.0"
echo "  module load PyYAML/6.0.1-GCCcore-13.2.0"
echo "  source ~/envs/yolo-hipster/bin/activate"
echo ""
