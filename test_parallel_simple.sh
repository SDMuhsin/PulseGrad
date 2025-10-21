#!/bin/bash

# Simple test with 3 parallel instances to verify thread-safety
echo "Starting simple thread-safety test with 3 parallel instances..."
echo "Each instance will run MNIST, ResNet18 for 1 epoch with k_folds=2"
echo ""

# Clean up previous test results
rm -f ./results/classification_results.csv

# Run 3 parallel instances with different optimizers
echo "Launching instance 1/3 with optimizer: adam"
python3 src/train_imageclassification.py \
    --dataset MNIST \
    --model resnet18 \
    --optimizer adam \
    --epochs 1 \
    --batch_size 64 \
    --lr 0.001 \
    --k_folds 2 \
    > "./results/test_log_adam.txt" 2>&1 &

echo "Launching instance 2/3 with optimizer: sgd"
python3 src/train_imageclassification.py \
    --dataset MNIST \
    --model resnet18 \
    --optimizer sgd \
    --epochs 1 \
    --batch_size 64 \
    --lr 0.001 \
    --k_folds 2 \
    > "./results/test_log_sgd.txt" 2>&1 &

echo "Launching instance 3/3 with optimizer: adamw"
python3 src/train_imageclassification.py \
    --dataset MNIST \
    --model resnet18 \
    --optimizer adamw \
    --epochs 1 \
    --batch_size 64 \
    --lr 0.001 \
    --k_folds 2 \
    > "./results/test_log_adamw.txt" 2>&1 &

# Wait for all background jobs to complete
echo ""
echo "Waiting for all instances to complete..."
wait

echo ""
echo "All instances completed!"
echo ""

# Check results
if [ -f "./results/classification_results.csv" ]; then
    echo "✓ CSV file created successfully"
    line_count=$(wc -l < ./results/classification_results.csv)
    echo "  Total lines: $line_count (expected 4: 1 header + 3 data rows)"

    echo ""
    echo "CSV contents:"
    cat ./results/classification_results.csv

    echo ""
    # Count unique optimizer entries
    unique_count=$(tail -n +2 ./results/classification_results.csv | cut -d',' -f4 | sort -u | wc -l)
    echo "  Unique optimizers: $unique_count (expected 3)"

    if [ "$line_count" -eq 4 ] && [ "$unique_count" -eq 3 ]; then
        echo ""
        echo "✓ Thread-safety test PASSED!"
    else
        echo ""
        echo "✗ Thread-safety test FAILED - unexpected line count or duplicates"
    fi
else
    echo "✗ ERROR: CSV file not created!"
fi

echo ""
echo "Check individual logs for errors:"
echo "  tail -n 20 ./results/test_log_*.txt"
