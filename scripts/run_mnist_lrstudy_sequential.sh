#!/bin/bash
# Run MNIST LR studies sequentially to avoid resource contention

export CUDA_VISIBLE_DEVICES=1

echo "=========================================="
echo "Starting AdaBelief LR Study + K-Fold CV"
echo "Start time: $(date)"
echo "=========================================="

python3 -u src/train_imageclassification.py \
    --dataset MNIST \
    --model resnet18 \
    --optimizer adabelief \
    --epochs 10 \
    --batch_size 64 \
    --lr_study \
    --k_folds 2 \
    2>&1 | tee mnist_adabelief_lrstudy_new.log

ADABELIEF_EXIT=$?

echo ""
echo "=========================================="
echo "AdaBelief completed with exit code: $ADABELIEF_EXIT"
echo "End time: $(date)"
echo "=========================================="
echo ""
sleep 5

echo "=========================================="
echo "Starting PulseAdaBeliefAdaptive LR Study + K-Fold CV"
echo "Start time: $(date)"
echo "=========================================="

python3 -u src/train_imageclassification.py \
    --dataset MNIST \
    --model resnet18 \
    --optimizer pulseadabelief_adaptive \
    --epochs 10 \
    --batch_size 64 \
    --lr_study \
    --k_folds 2 \
    2>&1 | tee mnist_pulseadabelief_lrstudy_new.log

PULSE_EXIT=$?

echo ""
echo "=========================================="
echo "PulseAdaBeliefAdaptive completed with exit code: $PULSE_EXIT"
echo "End time: $(date)"
echo "=========================================="
echo ""

echo "=========================================="
echo "BOTH EXPERIMENTS COMPLETE"
echo "=========================================="
echo "AdaBelief exit code: $ADABELIEF_EXIT"
echo "PulseAdaBeliefAdaptive exit code: $PULSE_EXIT"
echo ""
echo "Results saved to:"
echo "  - mnist_adabelief_lrstudy_new.log"
echo "  - mnist_pulseadabelief_lrstudy_new.log"
echo "  - results/classification_results.csv"
echo "  - results/lr_search_cache.json"
echo "=========================================="
