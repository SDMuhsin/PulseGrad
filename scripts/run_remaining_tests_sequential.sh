#!/bin/bash
# Run remaining tests sequentially on CUDA device 1

export CUDA_VISIBLE_DEVICES=1

echo "=== Starting Sequential Tests on CUDA:1 ==="
echo "Started at: $(date)"
echo ""

# Test 1: DiffGrad baseline
echo "[1/4] Running DiffGrad baseline..."
echo "Start time: $(date)"
python3 src/train_imageclassification.py --dataset CIFAR100 --model resnet18 --optimizer diffgrad --epochs 10 --batch_size 64 --lr 0.001 --k_folds 2
echo "Completed at: $(date)"
echo ""

# Test 2: PulseAdamAdaptive
echo "[2/4] Running PulseAdamAdaptive..."
echo "Start time: $(date)"
python3 src/train_imageclassification.py --dataset CIFAR100 --model resnet18 --optimizer pulseadam_adaptive --epochs 10 --batch_size 64 --lr 0.001 --k_folds 2
echo "Completed at: $(date)"
echo ""

# Test 3: PulseSGDAdaptive
echo "[3/4] Running PulseSGDAdaptive..."
echo "Start time: $(date)"
python3 src/train_imageclassification.py --dataset CIFAR100 --model resnet18 --optimizer pulsesgd_adaptive --epochs 10 --batch_size 64 --lr 0.001 --k_folds 2
echo "Completed at: $(date)"
echo ""

# Test 4: PulseDiffGradAdaptive
echo "[4/4] Running PulseDiffGradAdaptive..."
echo "Start time: $(date)"
python3 src/train_imageclassification.py --dataset CIFAR100 --model resnet18 --optimizer pulsediffgrad_adaptive --epochs 10 --batch_size 64 --lr 0.001 --k_folds 2
echo "Completed at: $(date)"
echo ""

echo "=== All Tests Complete ==="
echo "Finished at: $(date)"
echo ""
echo "Final results:"
tail -5 results/classification_results.csv
