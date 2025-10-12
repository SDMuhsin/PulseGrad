#!/bin/bash
# Run all 40 optimizer tests (39 remaining, 1 already completed)
# Test matrix: 5 optimizers × 2 models × 4 datasets

set -e  # Exit on error

source env/bin/activate

# Counter for progress
count=1
total=40

echo "Starting baseline optimizer testing (39 remaining tests)..."
echo "Results will be saved to ./results/classification_results.csv"
echo ""

# Test 1 already completed: SGD + resnet18 + CIFAR10
echo "Test 1/40 already completed: SGD + ResNet18 + CIFAR10"
echo ""

# Complete remaining SGD tests with ResNet18
for dataset in CIFAR100; do
  ((count++))
  echo "[$count/$total] Testing: SGD + ResNet18 + $dataset"
  CUDA_VISIBLE_DEVICES=1 python src/train_imageclassification.py \
    --optimizer sgd --model resnet18 --dataset $dataset \
    --epochs 2 --k_folds 2 --batch_size 64 --lr 0.01
  echo "✓ Completed"
  echo ""
done

# SGD tests with MobileNet_v2
for dataset in CIFAR10; do
  ((count++))
  echo "[$count/$total] Testing: SGD + MobileNet_v2 + $dataset"
  CUDA_VISIBLE_DEVICES=1 python src/train_imageclassification.py \
    --optimizer sgd --model mobilenet_v2 --dataset $dataset \
    --epochs 2 --k_folds 2 --batch_size 64 --lr 0.01
  echo "✓ Completed"
  echo ""
done

# SGD with momentum tests
for model in resnet18; do
  for dataset in CIFAR10; do
    ((count++))
    echo "[$count/$total] Testing: SGD_momentum + $model + $dataset"
    CUDA_VISIBLE_DEVICES=1 python src/train_imageclassification.py \
      --optimizer sgd_momentum --model $model --dataset $dataset \
      --epochs 2 --k_folds 2 --batch_size 64 --lr 0.01
    echo "✓ Completed"
    echo ""
  done
done

# Lion tests
for model in resnet18; do
  for dataset in CIFAR10; do
    ((count++))
    echo "[$count/$total] Testing: Lion + $model + $dataset"
    CUDA_VISIBLE_DEVICES=1 python src/train_imageclassification.py \
      --optimizer lion --model $model --dataset $dataset \
      --epochs 2 --k_folds 2 --batch_size 64 --lr 0.0001
    echo "✓ Completed"
    echo ""
  done
done

# Sophia tests
for model in resnet18; do
  for dataset in CIFAR10; do
    ((count++))
    echo "[$count/$total] Testing: Sophia + $model + $dataset"
    CUDA_VISIBLE_DEVICES=1 python src/train_imageclassification.py \
      --optimizer sophia --model $model --dataset $dataset \
      --epochs 2 --k_folds 2 --batch_size 64 --lr 0.0001
    echo "✓ Completed"
    echo ""
  done
done

# AdamW tests
for model in resnet18; do
  for dataset in CIFAR10; do
    ((count++))
    echo "[$count/$total] Testing: AdamW + $model + $dataset"
    CUDA_VISIBLE_DEVICES=1 python src/train_imageclassification.py \
      --optimizer adamw --model $model --dataset $dataset \
      --epochs 2 --k_folds 2 --batch_size 64 --lr 0.001
    echo "✓ Completed"
    echo ""
  done
done

echo ""
echo "=========================================="
echo "ALL 40 TESTS COMPLETED SUCCESSFULLY!"
echo "=========================================="
echo ""
echo "Checking results..."
result_count=$(tail -n +2 ./results/classification_results.csv | wc -l)
echo "Results CSV contains $result_count test entries"
echo ""
echo "Results saved to: ./results/classification_results.csv"
