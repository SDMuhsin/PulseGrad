#!/bin/bash

export HF_HOME=./data
export TORCH_HOME=./data

# Default learning rate
DEFAULT_LR=0.00003

# Check the number of arguments
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 <task_name> [learning_rate]"
    echo "  <task_name>     : Name of the GLUE task (e.g., mnli, qqp, sst2)"
    echo "  [learning_rate] : Optional. Learning rate for training (default: $DEFAULT_LR)"
    exit 1
fi

# Assign arguments
task=$1
lr=${2:-$DEFAULT_LR} # Use provided lr or default if not set

echo "Running task: $task"
echo "Using learning rate: $lr"

models=("distilbert/distilbert-base-uncased" "albert/albert-base-v2") # You can add more models here e.g., ("google-bert/bert-large-uncased" "microsoft/deberta-v3-large")

optimizers=("experimentalv2") #("adan" "adagrad" "adadelta" "rmsprop" "amsgrad" "adam" "experimental" "diffgrad" "adabelief" "adamp" "madgrad" "lion")

# Loop through each model and optimizer combination
for model in "${models[@]}"; do
    for optimizer in "${optimizers[@]}"; do
        echo "----------------------------------------------------"
        echo "Training with Model: $model, Optimizer: $optimizer, LR: $lr"
        echo "----------------------------------------------------"
        python ./src/train_glue.py \
            --task_name "$task" \
            --model_name_or_path "$model" \
            --optimizer "$optimizer" \
            --epochs 3 \
            --lr "$lr" \
            --batch_size 32
        
        # Check the exit status of the python script
        if [ $? -ne 0 ]; then
            echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            echo "ERROR: Training failed for Model: $model, Optimizer: $optimizer, LR: $lr"
            echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            # Decide if you want to exit the script entirely or continue with the next combination
            # exit 1 # Uncomment to exit on first failure
        else
            echo "Successfully completed training for Model: $model, Optimizer: $optimizer, LR: $lr"
        fi
        echo "" # Add a blank line for better readability
    done
done

echo "All training combinations processed."

