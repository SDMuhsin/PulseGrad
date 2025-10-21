#!/bin/bash

# Test script to verify thread-safe file operations with parallel instances
# This will run 10 parallel instances simultaneously to test for race conditions

echo "Starting thread-safety test with 10 parallel instances..."
echo "Each instance will run MNIST, ResNet18 for 1 epoch with k_folds=2"
echo ""

# Clean up test files
rm -f ./results/test_*.csv ./results/test_lr_cache.json

# Define test configurations - different optimizers to create variety
optimizers=(
    "adam"
    "sgd"
    "adamw"
    "rmsprop"
    "adadelta"
    "adabelief"
    "lion"
    "adan"
    "madgrad"
    "adamp"
)

# Run 10 parallel instances with different optimizers
for i in {0..9}; do
    opt="${optimizers[$i]}"
    echo "Launching instance $((i+1))/10 with optimizer: $opt"

    # Run in background with unique log file
    python src/train_imageclassification.py \
        --dataset MNIST \
        --model resnet18 \
        --optimizer "$opt" \
        --epochs 1 \
        --batch_size 64 \
        --lr 0.001 \
        --k_folds 2 \
        > "./results/test_log_${opt}.txt" 2>&1 &
done

# Wait for all background jobs to complete
echo ""
echo "Waiting for all instances to complete..."
wait

echo ""
echo "All instances completed!"
echo ""
echo "Checking results..."

# Check CSV file
if [ -f "./results/classification_results.csv" ]; then
    echo "CSV file created successfully:"
    line_count=$(wc -l < ./results/classification_results.csv)
    echo "  Total lines: $line_count (expected 11: 1 header + 10 data rows)"

    # Check for duplicate entries
    echo ""
    echo "First 5 lines of CSV:"
    head -n 5 ./results/classification_results.csv

    echo ""
    echo "Checking for duplicates..."
    # Count unique optimizer entries (should be 10)
    unique_count=$(tail -n +2 ./results/classification_results.csv | cut -d',' -f4 | sort -u | wc -l)
    echo "  Unique optimizers in CSV: $unique_count (expected 10)"
else
    echo "ERROR: CSV file not created!"
fi

echo ""
echo "Thread-safety test complete!"
echo ""
echo "To manually inspect results:"
echo "  CSV: cat ./results/classification_results.csv"
echo "  Logs: ls -la ./results/test_log_*.txt"
