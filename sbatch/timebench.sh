#!/bin/bash
echo "Beginning timebench.py sbatch script submissions."


sbatch \
--nodes=1 \
--ntasks-per-node=1 \
--cpus-per-task=1 \
--gpus=1 \
--mem=4000M \
--time=0-01:00 \
--chdir=/scratch/sdmuhsin/DiffGrad2 \
--output=timebench-%N-%j.out \
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
    python3 src/timebench.py --devices=cuda
"

echo "All timing jobs submitted."

