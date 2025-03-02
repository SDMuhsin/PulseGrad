#!/bin/bash

echo "Beginning train_imageclassification.py sbatch script submissions."

# Define dataset-model pairs based on task difficulty and model size
declare -A pairs
pairs["MNIST"]="squeezenet1_0"
#pairs["FMNIST"]="resnet50"
#pairs["CIFAR10"]="vgg16"
#pairs["CIFAR100"]="vit_b_16"
#pairs["STL10"]="efficientnet_v2_s"

# List of optimizers to loop through
optimizers=(adagrad adadelta rmsprop amsgrad adam experimental diffgrad)

# Loop through each dataset-model pair
for dataset in "${!pairs[@]}"; do
    model="${pairs[$dataset]}"
    
    # Remove slash from model name if present (for output filename safety)
    model_filename=${model//\//}

    # Loop through optimizers for the current dataset-model pair
    for optimizer in "${optimizers[@]}"; do
        echo "Submitting job with dataset: $dataset, model: $model, optimizer: $optimizer"
        sbatch \
            --nodes=1 \
            --ntasks-per-node=1 \
            --cpus-per-task=1 \
            --gpus=1 \
            --mem=8000M \
            --time=3-00:00 \
            --chdir=/scratch/sdmuhsin/DiffGrad2 \
            --output=${optimizer}-${model_filename}-${dataset}-%N-%j.out \
            --wrap="
                export TRANSFORMERS_CACHE=\"./downloads\"
                export HF_HOME=\"./downloads\"
                export TORCH_HOME=\"./downloads\"
                module load python/3.10
                module load arrow/16.1.0
                source ./env/bin/activate
                echo 'Environment loaded'
                which python3
                export PYTHONPATH=\"\$PYTHONPATH:\$(pwd)\"
                python3 src/train_imageclassification.py --dataset=$dataset --model=$model --optimizer=$optimizer --epochs=100 --batch_size=64 --lr=0.001
            "
    done
done

echo "All jobs submitted."

