#!/bin/bash

echo "Beginning train_glue.py sbatch script submissions."

# Define the learning rates to iterate through
learning_rates=("0.0001" "2e-5")

for lr in "${learning_rates[@]}"; do
    echo "----------------------------------------------------"
    echo "Processing with Learning Rate: $lr"
    echo "----------------------------------------------------"
    for dataset in mrpc rte cola stsb qnli sst2; do
        case $dataset in
            mrpc|rte|stsb|cola)
                time_req="1-00:00"
                ;;
            sst2|qnli)
                time_req="4-00:00"
                ;;
            *)
                time_req="7-00:00" # Default for any other datasets, though current loop is fixed
                ;;
        esac

        echo "Submitting job for dataset: $dataset, learning rate: $lr, time: $time_req"
        sbatch \
            --nodes=1 \
            --ntasks-per-node=1 \
            --cpus-per-task=1 \
            --gpus=1 \
            --mem=16000M \
            --time=$time_req \
            --chdir=/scratch/sdmuhsin/DiffGrad2 \
            --output=${dataset}-lr${lr}-%N-%j.out \
            --wrap="
                export TRANSFORMERS_CACHE=\"./data\"
                export HF_HOME=\"./data\"
                export TORCH_HOME=\"./data\"
                module load python/3.10
                module load arrow/16.1.0
                source ./env/bin/activate
                echo 'Environment loaded'
                which python3
                export PYTHONPATH=\"\$PYTHONPATH:\$(pwd)\"
                echo 'Executing ./scripts/run_glue.sh $dataset $lr'
                ./scripts/run_glue.sh $dataset $lr
            "
        # Check if sbatch submission was successful (optional, but good practice)
        if [ $? -ne 0 ]; then
            echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            echo "ERROR: sbatch submission failed for dataset: $dataset, LR: $lr"
            echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        else
            echo "Successfully submitted job for dataset: $dataset, LR: $lr"
        fi
        echo "" # Add a blank line for better readability between submissions
    done
done

echo "All jobs submitted."

