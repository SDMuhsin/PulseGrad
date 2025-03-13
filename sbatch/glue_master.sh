#!/bin/bash

echo "Beginning train_glue.py sbatch script submissions."


for dataset in mrpc; do # rte cola stsb qnli qqp mnli sst2; do
    case $dataset in
	mrpc|rte|stsb|cola)
	    time_req="1-00:00"
	    ;;
	sst2)
	    time_req="4-00:00"
	    ;;
	*)
	    time_req="7-00:00"
	    ;;
    esac

    echo "Submitting job with dataset: $dataset and time: $time_req"
    sbatch \
	--nodes=1 \
	--ntasks-per-node=1 \
	--cpus-per-task=1 \
	--gpus=1 \
	--mem=16000M \
	--time=$time_req \
	--chdir=/scratch/sdmuhsin/DiffGrad2 \
	--output=${dataset}-%N-%j.out \
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
	    ./scripts/run_glue.sh $dataset
	"
done

echo "All jobs submitted."

