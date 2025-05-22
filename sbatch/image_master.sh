!/bin/bash

echo "Beginning train_imageclassification.py sbatch script submissions."

# Define dataset-model pairs based on task difficulty and model size
declare -A pairs
pairs["MNIST"]="squeezenet1_0"
pairs["FMNIST"]="resnet50"
pairs["CIFAR10"]="densenet121"
pairs["CIFAR100"]="vit_b_16"
#pairs["STL10"]="efficientnet_v2_s"

# Define time allocations for each dataset
declare -A dataset_times
dataset_times["MNIST"]="1-04:00:00"  # 1 day 4 hours
dataset_times["FMNIST"]="7-00:00:00" # Default for FMNIST, can be adjusted
dataset_times["CIFAR10"]="2-00:00:00" # 2 days
dataset_times["CIFAR100"]="3-04:00:00" # 3 days 4 hours
#dataset_times["STL10"]="7-00:00:00" # Default for STL10, if uncommented

# List of optimizers to loop through
optimizers=( adabelief adamp madgrad adan lion ) #adagrad adadelta rmsprop amsgrad adam experimental diffgrad)

# Loop through each dataset-model pair
for dataset in "${!pairs[@]}"; do
    model="${pairs[$dataset]}"
    job_time="${dataset_times[$dataset]}" # Get the time for the current dataset

    # Remove slash from model name if present (for output filename safety)
    model_filename=${model//\//}

    # Loop through optimizers for the current dataset-model pair
    for optimizer in "${optimizers[@]}"; do
        echo "Submitting job with dataset: $dataset, model: $model, optimizer: $optimizer, time: $job_time"
        sbatch \
            --nodes=1 \
            --ntasks-per-node=1 \
            --cpus-per-task=1 \
            --gpus=1 \
            --mem=8000M \
            --time=$job_time \
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
                python3 src/train_imageclassification.py --dataset=$dataset --model=$model --optimizer=$optimizer --epochs=100 --batch_size=64 --lr=0.0001
            "
    done
done

echo "All jobs submitted."
