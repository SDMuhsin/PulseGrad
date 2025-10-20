#!/bin/bash
# Test all new adaptive pulse optimizers for 1 epoch to check for bugs

echo "Testing all new adaptive pulse optimizers..."
echo "Using CIFAR100, ResNet18, 1 epoch, 2 folds"
echo ""

export CUDA_VISIBLE_DEVICES=1

optimizers=(
    "pulseadadelta_adaptive"
    "pulsermsprop_adaptive"
    "pulseamsgrad_adaptive"
    "pulseadamw_adaptive"
    "pulseadabelief_adaptive"
    "pulseadamp_adaptive"
    "pulsemadgrad_adaptive"
    "pulseadan_adaptive"
)

for opt in "${optimizers[@]}"; do
    echo "================================"
    echo "Testing: $opt"
    echo "================================"
    python3 src/train_imageclassification.py --dataset CIFAR100 --model resnet18 --optimizer $opt --epochs 1 --batch_size 64 --lr 0.001 --k_folds 2 2>&1 | tail -20

    if [ $? -eq 0 ]; then
        echo "✓ $opt: SUCCESS"
    else
        echo "✗ $opt: FAILED"
    fi
    echo ""
done

echo "================================"
echo "All tests complete!"
echo "================================"
