#!/bin/bash

# Quick script to check experiment results

echo "========================================"
echo "Experiment Status Check"
echo "========================================"
echo ""

# Check running processes
RUNNING=$(ps aux | grep "train_imageclassification" | grep -v grep | wc -l)
echo "Running experiments: $RUNNING"
echo ""

# Show latest results from CSV
echo "Latest results (MNIST, resnet18, 10 epochs, k_folds=2):"
echo "--------------------------------------------------------"
grep "MNIST,resnet18" results/classification_results.csv | \
  grep "10,64" | \
  awk -F',' '{printf "%-20s | Acc: %s%% ± %s%% | Optimizer: %s\n", $1, $9, $10, $4}' | \
  tail -5

echo ""
echo "Target to beat:"
echo "  Sophia: 99.4500% ± 0.0133%"
echo "  Lion:   99.4283% ± 0.0250%"
echo ""
