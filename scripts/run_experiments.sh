#!/bin/bash

# Experiment script for PulseGrad enhancement research
# Run this when GPU memory is available

set -e

echo "========================================"
echo "PulseGrad Enhancement Experiments"
echo "========================================"

# Activate environment
source env/bin/activate

# Test both H3 and H2 implementations
echo ""
echo "[1/2] Running H3: Scaled Friction Coefficient (alpha=10)..."
CUDA_VISIBLE_DEVICES=1 python3 src/train_imageclassification.py \
    --dataset MNIST \
    --model resnet18 \
    --optimizer pulsegrad_enhanced \
    --epochs 10 \
    --batch_size 64 \
    --lr 0.01 \
    --k_folds 5 \
    2>&1 | tee results/h3_alpha10.log

echo ""
echo "[2/2] Running H2: Smoothed Pulse (EMA)..."
CUDA_VISIBLE_DEVICES=1 python3 src/train_imageclassification.py \
    --dataset MNIST \
    --model resnet18 \
    --optimizer pulsegrad_h2 \
    --epochs 10 \
    --batch_size 64 \
    --lr 0.01 \
    --k_folds 5 \
    2>&1 | tee results/h2_smoothed.log

echo ""
echo "========================================"
echo "Experiments complete!"
echo "========================================"
echo ""
echo "Results comparison:"
echo "Baseline (PulseGrad): 99.4967% ± 0.0215%"
echo "Target (Lion):        99.5233% ± 0.0714%"
echo "Target (Sophia):      99.5167% ± 0.0450%"
echo ""
echo "Check the last line of results/classification_results.csv for results"
tail -2 results/classification_results.csv
