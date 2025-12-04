!/bin/bash

echo "Beginning train_imageclassification.py sbatch script submissions."

# Define dataset-model pairs based on task difficulty and model size
declare -A pairs
#pairs["MNIST"]="squeezenet1_0"
#pairs["FMNIST"]="resnet50"
pairs["CIFAR10"]="densenet121"
pairs["CIFAR100"]="vit_b_16"
#pairs["STL10"]="efficientnet_v2_s"

# Define time allocations for each dataset
declare -A dataset_times
dataset_times["MNIST"]="2-00:00:00"  # 1 day 4 hours
dataset_times["FMNIST"]="2-00:00:00" # Default for FMNIST, can be adjusted
dataset_times["CIFAR10"]="7-00:00:00" # 7 days, 
dataset_times["CIFAR100"]="7-00:00:00" # 3 days 4 hours
#dataset_times["STL10"]="7-00:00:00" # Default for STL10, if uncommented

# List of optimizers to loop through
optimizers=(
    # Baseline optimizers
    #adagrad adadelta rmsprop amsgrad adam adabelief adamp madgrad adan lion sgd sgd_momentum adamw diffgrad sophia
    # Pulse adaptive optimizers
    #pulseadam_adaptive pulsesgd_adaptive pulsediffgrad_adaptive pulselion_adaptive pulsesophia_adaptive
    #pulseadadelta_adaptive pulsermsprop_adaptive pulseamsgrad_adaptive pulseadamw_adaptive
    #pulseadabelief_adaptive pulseadamp_adaptive pulsemadgrad_adaptive pulseadan_adaptive
    dcp_sgd
)

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
            --cpus-per-task=2 \
            --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1 \
            --mem=16000M \
            --time=$job_time \
            --output=./logs/${optimizer}-${model_filename}-${dataset}-%N-%j.out \
            --wrap="
                export TRANSFORMERS_CACHE=\"./cache\"
                export HF_HOME=\"./cache\"
                export TORCH_HOME=\"./cache\"
                module load arrow
                source ./env/bin/activate
                echo 'Environment loaded'
                which python3
                export PYTHONPATH=\"\$PYTHONPATH:\$(pwd)\"
                python3 src/train_imageclassification.py --dataset=$dataset --model=$model --optimizer=$optimizer --epochs=100 --batch_size=64 --lr=0.001 --lr_study
            "
    done
done

echo "All jobs submitted."
