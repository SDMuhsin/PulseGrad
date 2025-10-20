#!/bin/bash
# Script to validate adaptive pulse across multiple optimizer families

echo "=== Adaptive Pulse Cross-Family Validation ==="
echo "Testing on CIFAR100, ResNet18, 10 epochs, 2-fold CV"
echo ""

# Baselines
echo "[1/6] Running Adam baseline..."
python3 src/train_imageclassification.py --dataset CIFAR100 --model resnet18 --optimizer adam --epochs 10 --batch_size 64 --lr 0.001 --k_folds 2

echo "[2/6] Running SGD baseline..."
python3 src/train_imageclassification.py --dataset CIFAR100 --model resnet18 --optimizer sgd_momentum --epochs 10 --batch_size 64 --lr 0.001 --k_folds 2

echo "[3/6] Running DiffGrad baseline..."
python3 src/train_imageclassification.py --dataset CIFAR100 --model resnet18 --optimizer diffgrad --epochs 10 --batch_size 64 --lr 0.001 --k_folds 2

# Adaptive variants
echo "[4/6] Running PulseAdamAdaptive..."
python3 src/train_imageclassification.py --dataset CIFAR100 --model resnet18 --optimizer pulseadam_adaptive --epochs 10 --batch_size 64 --lr 0.001 --k_folds 2

echo "[5/6] Running PulseSGDAdaptive..."
python3 src/train_imageclassification.py --dataset CIFAR100 --model resnet18 --optimizer pulsesgd_adaptive --epochs 10 --batch_size 64 --lr 0.001 --k_folds 2

echo "[6/6] Running PulseDiffGradAdaptive..."
python3 src/train_imageclassification.py --dataset CIFAR100 --model resnet18 --optimizer pulsediffgrad_adaptive --epochs 10 --batch_size 64 --lr 0.001 --k_folds 2

echo ""
echo "=== All tests complete! ==="
echo "Results saved to results/classification_results.csv"
