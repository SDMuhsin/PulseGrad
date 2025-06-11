# PulseGrad: Robust Neural Network Training with Pulse-Aware Moment Estimates

This repository contains the official code for the paper, "PulseGrad: Robust Neural Network Training with Pulse‑Aware Moment Estimates".

## Abstract

*PulseGrad* is an adaptive optimizer that couples first-order momentum with a pulse-aware second-moment estimate. Whereas existing algorithms either depend exclusively on gradient magnitudes (Adam, RMSProp) or adjust learning rates through instantaneous gradient differences (DiffGrad), PulseGrad blends the two signals in a single, coherent update. Its variance accumulator supplements the squared gradient with a historical *pulse* term—the squared difference between successive gradients—so that both local curvature and temporal variability are encoded. We prove that this design bounds the growth of the variance, promotes stable updates, and preserves convergence guarantees that match the strongest results for Adam-type methods. Comprehensive experiments on vision (CIFAR-10/100, FMNIST, MNIST) confirm that PulseGrad achieves competitive or superior accuracy compared to state-of-the-art optimizers.

## The PulseGrad Algorithm

PulseGrad modifies the second-moment estimate (<span class="math-inline">v\_t</span>) of Adam-like optimizers by incorporating a "pulse" term, <span class="math-inline">\\gamma \\Delta g\_t^2</span>, which is the squared difference between successive gradients (<span class="math-inline">g\_t</span> and <span class="math-inline">g\_\{t\-1\}</span>). This makes the variance estimate sensitive to both gradient magnitude and temporal volatility.

The core update for the second moment is:

<span class="math-block">v\_t \= \\beta\_2 v\_\{t\-1\} \+ \(1\-\\beta\_2\)\\bigl\(g\_t^\{2\}\+\\gamma\\,\\Delta g\_t^\{2\}\\bigr\)</span>

The full algorithm also incorporates bias correction and a friction term to modulate the update size, as detailed in the paper.

## Repository Structure

This repository is organized as follows:

* `src/experimental/exp.py`: Contains the implementation of the **PulseGrad** optimizer.
* `src/train_imageclassification.py`: The main script for running image classification experiments on MNIST, FMNIST, CIFAR-10, and CIFAR-100.
* `src/ablation.py`: Script to reproduce the ablation studies on hyperparameters like <span class="math-inline">\\beta\_1, \\beta\_2, \\gamma</span>, etc.
* `src/tabulate_imageclass.py`: A utility script to process raw experimental outputs and generate the summary tables found in the paper.
* `results/`: The default directory where raw results (CSVs, JSONs, and plots) from experiments are stored.
* `install.sh`: A shell script to install all necessary dependencies.

## Requirements and Installation

To set up the environment and install the required packages, run the provided installation script:

    bash install.sh

## Reproducing Experiments

You can run the image classification experiments using the `train_imageclassification.py` script. Specify the dataset, model, and optimizer as arguments. For example, to run PulseGrad on CIFAR-10 with a DenseNet-121 model, as described in the paper:

    python src/train_imageclassification.py \
        --dataset CIFAR10 \
        --model DenseNet-121 \
        --optimizer pulsegrad \
        --lr 1e-3

To view a formatted table of the results, run the tabulation script after your experiments are complete:

    python src/tabulate_imageclass.py

## Citation

If you find this work useful in your research, please consider citing our paper. We currently do not have a arxiv upload, but will provide one on request to sdmuhsin@gmail.com
