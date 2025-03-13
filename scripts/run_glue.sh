#!/bin/bash

export HF_HOME=./data
export TORCH_HOME=./data

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <task_name>"
    exit 1
fi

task=$1

models=("bert-base-uncased" "distilbert-base-uncased")
optimizers=("adagrad" "adadelta" "rmsprop" "amsgrad" "adam" "experimental" "diffgrad")

for model in "${models[@]}"; do
    for optimizer in "${optimizers[@]}"; do
        python ./src/train_glue.py \
            --task_name "$task" \
            --model_name_or_path "$model" \
            --optimizer "$optimizer" \
            --epochs 5 \
            --lr 5e-5 \
            --batch_size 32
    done
done
