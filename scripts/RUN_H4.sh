#!/bin/bash
# Run H4 experiment: PulseGrad with weight decay

echo "=========================================="
echo "Running H4: PulseGrad + Weight Decay"
echo "=========================================="
echo ""
echo "Target: Beat 99.4500% (Sophia)"
echo "Baseline: 99.4433% (PulseGrad)"
echo "Change: weight_decay=1e-2 (was 0)"
echo ""

CUDA_VISIBLE_DEVICES=1 python3 ./src/train_imageclassification.py \
    --dataset=MNIST \
    --model=resnet18 \
    --optimizer=pulsegrad_wd \
    --batch_size=64 \
    --lr=0.01 \
    --epochs=10 \
    --k_folds=2

echo ""
echo "=========================================="
echo "Results:"
echo "=========================================="
tail -1 results/classification_results.csv | awk -F',' '{print "Accuracy: " $9 " ± " $10}'
echo ""
echo "Compare to:"
echo "  Sophia:  99.4500% ± 0.0133%"
echo "  Baseline: 99.4433% ± 0.0333%"
echo ""
