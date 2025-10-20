#!/bin/bash
# PulseGrad Installation Script
# Installs all required dependencies for running experiments

set -e  # Exit on error

echo "Installing PulseGrad dependencies..."
echo ""

# Core PyTorch and ML libraries
echo "=== Installing PyTorch and core ML libraries ==="
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Scientific computing and data processing
echo ""
echo "=== Installing scientific computing libraries ==="
python3 -m pip install numpy pandas matplotlib scikit-learn seaborn

# PyTorch utilities
echo ""
echo "=== Installing PyTorch utilities ==="
python3 -m pip install torchinfo torchmetrics tabulate

# Hugging Face and transformers ecosystem
echo ""
echo "=== Installing transformers and related packages ==="
python3 -m pip install transformers evaluate datasets accelerate sentencepiece protobuf

# Third-party optimizers
echo ""
echo "=== Installing third-party optimizer packages ==="
python3 -m pip install adabelief-pytorch      # AdaBelief optimizer
python3 -m pip install adamp                  # AdamP optimizer (includes SGDP)
python3 -m pip install madgrad                # MADGRAD optimizer
python3 -m pip install adan-pytorch           # Adan optimizer
python3 -m pip install lion-pytorch           # Lion optimizer
python3 -m pip install sophia-optimizer       # Sophia-G optimizer
python3 -m pip install torch-optimizer        # Collection of optimizers (DiffGrad, etc.)

echo ""
echo "✓ Installation complete!"
echo ""
echo "Installed optimizers:"
pip list | grep -iE "sophia|adabelief|adamp|madgrad|adan|lion|torch-optimizer"
