#!/bin/bash
echo "Beginning ablation.py sbatch script submissions."

# Hyper‑parameters to ablate
ablations=(lr beta1 beta2 gamma eps)

# Fixed dataset / model for this study
dataset="CIFAR10"
model="resnet18"

for ablation in "${ablations[@]}"; do
    echo "Submitting job for ablation: $ablation"
    sbatch \
        --nodes=1 \
        --ntasks-per-node=1 \
        --cpus-per-task=1 \
        --gpus=1 \
        --mem=8000M \
        --time=3-00:00 \
        --chdir=/scratch/sdmuhsin/DiffGrad2 \
        --output=ablation-${ablation}-${model}-${dataset}-%N-%j.out \
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
            python3 src/ablation.py --ablation=$ablation --dataset=$dataset --model=$model --epochs=10 --plot
        "
done

echo "All ablation jobs submitted."

