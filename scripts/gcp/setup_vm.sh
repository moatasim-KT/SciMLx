#!/bin/bash
# GCP Compute Engine Deep Learning VM Setup Script for SciMLx
set -e

echo "--- Initializing SciMLx CUDA Setup ---"

# 1. Install uv
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
fi

# 2. Check for GPU and CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
    echo "WARNING: No GPU detected. Please ensure you attached a GPU to this instance."
fi

# 3. Create virtual environment and install dependencies
echo "Setting up environment..."
uv venv .venv
source .venv/bin/activate
uv pip install -r pyproject.toml

# 4. Verify PyTorch CUDA
echo "Verifying PyTorch CUDA availability..."
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo "--- Setup Complete ---"
echo "To run a benchmark: python train.py --benchmark burgers_1d"
