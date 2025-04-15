#!/bin/bash

export HF_HOME=./data
export TORCH_HOME=./data

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <task_name>"
    exit 1
fi

task=$1

# Five similar transformer models
models=("xlnet-base-cased" "distilbert-base-uncased" "bert-base-uncased" "roberta-base"  "albert-base-v2")

# List of optimizers to iterate over
optimizers=("diffgrad" "experimental")

for model in "${models[@]}"; do
    for optimizer in "${optimizers[@]}"; do
        echo "Training with model: $model using optimizer: $optimizer"
        python ./src/train_glue.py \
            --task_name "$task" \
            --model_name_or_path "$model" \
            --optimizer "$optimizer" \
            --epochs 3 \
            --lr 0.0001 \
            --batch_size 32
    done
done
