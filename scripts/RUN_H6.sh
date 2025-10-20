#!/bin/bash
# Run H6: PulseGrad with Safe Zone Boost

echo "=========================================="
echo "H6: Safe Zone Boost (Intelligent Acceleration)"
echo "=========================================="
echo ""
echo "Change: Add boost = exp(-5*diff) to momentum"
echo "Effect: 2x momentum when diff≈0 (safe zones)"
echo "        1x momentum when diff>0.5 (risky zones)"
echo ""
echo "Target: Beat 99.4500% (Sophia)"
echo "Baseline: 99.4433% (PulseGrad)"
echo ""

CUDA_VISIBLE_DEVICES=1 python3 ./src/train_imageclassification.py \
    --dataset=MNIST \
    --model=resnet18 \
    --optimizer=pulsegrad_boost \
    --batch_size=64 \
    --lr=0.01 \
    --epochs=10 \
    --k_folds=2

echo ""
echo "=========================================="
echo "Results:"
tail -1 results/classification_results.csv | awk -F',' '{print "Accuracy: " $9 " ± " $10}'
echo ""
echo "Compare to:"
echo "  Sophia:   99.4500% ± 0.0133%"
echo "  Baseline: 99.4433% ± 0.0333%"
echo "=========================================="
