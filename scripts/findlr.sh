#!/bin/bash

export HF_HOME=./data
export TORCH_HOME=./data

# Task is fixed to 'rte'
task="rte"

# Model is fixed as per the original script's example
models=("distilbert/distilbert-base-uncased")

# Optimizers are fixed to 'experimental' and 'diffgrad'
optimizers=("experimental" "diffgrad")

# Number of uniform steps for learning rate iteration
num_lr_steps=10

echo "Starting hyperparameter sweep for task: $task"
echo "Models: ${models[*]}"
echo "Optimizers: ${optimizers[*]}"
echo "Number of learning rate steps per range: $num_lr_steps"
echo "Epochs: 3, Batch Size: 32"
echo "---"

for model in "${models[@]}"; do
    for optimizer in "${optimizers[@]}"; do
        echo ""
        echo "Training with Model: $model, Optimizer: $optimizer"
        echo "-----------------------------------------------------"

        # Learning rates from 0.001 to 0.0001 in 10 uniform steps
        echo "Iterating learning rates from 0.001 to 0.0001..."
        start_lr1=0.001
        end_lr1=0.0001
        for i in $(seq 0 $((num_lr_steps - 1))); do
            # Calculate learning rate using awk for precision
            # Formula: lr = start_lr + i * (end_lr - start_lr) / (total_steps - 1)
            current_lr=$(awk -v start="$start_lr1" -v end="$end_lr1" -v idx="$i" -v steps="$num_lr_steps" 'BEGIN { printf "%.8f", start + idx * (end - start) / (steps - 1) }')
            
            echo "Running with LR: $current_lr"
            python ./src/train_glue.py \
                --task_name "$task" \
                --model_name_or_path "$model" \
                --optimizer "$optimizer" \
                --epochs 3 \
                --lr "$current_lr" \
                --batch_size 32
            echo "---"
        done

        # Learning rates from 0.0001 to 0.000001 in 10 uniform steps
        echo "Iterating learning rates from 0.0001 to 0.000001..."
        start_lr2=0.0001
        end_lr2=0.000001
        for i in $(seq 0 $((num_lr_steps - 1))); do
            # Calculate learning rate using awk for precision
            current_lr=$(awk -v start="$start_lr2" -v end="$end_lr2" -v idx="$i" -v steps="$num_lr_steps" 'BEGIN { printf "%.8f", start + idx * (end - start) / (steps - 1) }')
            
            echo "Running with LR: $current_lr"
            python ./src/train_glue.py \
                --task_name "$task" \
                --model_name_or_path "$model" \
                --optimizer "$optimizer" \
                --epochs 3 \
                --lr "$current_lr" \
                --batch_size 32
            echo "---"
        done
        echo "-----------------------------------------------------"
    done
done

echo ""
echo "Script finished."
