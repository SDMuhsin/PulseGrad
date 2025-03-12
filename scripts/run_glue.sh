#!/bin/bash

export TASK_NAME=mrpc
export HF_HOME=./data
export TORCH_HOME=./data

python ./src/train_glue.py \
    --task_name mrpc \
    --model_name_or_path distilbert-base-uncased \
    --optimizer amsgrad \
    --epochs 3 \
    --lr 1e-5 \
    --batch_size 8
