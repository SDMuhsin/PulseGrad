import argparse
import random
import numpy as np
import os
import csv
import json
import hashlib
from datetime import datetime

# Configure caching directories for offline operation
# This ensures all model weights and downloads go to ./cache
os.environ['TORCH_HOME'] = os.path.join(os.getcwd(), 'cache')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.getcwd(), 'cache')
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'cache')

# Create cache directory if it doesn't exist
os.makedirs(os.environ['TORCH_HOME'], exist_ok=True)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

from adabelief_pytorch import AdaBelief
from adamp import AdamP                 # also exposes SGDP if you ever need it
import madgrad                         # madgrad.MADGRAD(...)
from adan_pytorch import Adan
from lion_pytorch import Lion
from sophia.sophia import SophiaG      # Sophia-G variant (direct import bypasses buggy __init__)

# Local experimental optimizers (ensure these modules are in your PYTHONPATH)
from experimental.lance import LANCE
from experimental.diffgrad import diffgrad
from experimental.cosmic import COSMIC
from experimental.fused_exp import Experimental
from experimental.exp2 import ExperimentalV2
from experimental.fused_exp_logged import ExperimentalLogged
from experimental.pulsegrad_enhanced import PulseGradEnhanced
from experimental.pulsegrad_h2 import PulseGradH2
from experimental.pulsegrad_wd import PulseGradWD
from experimental.pulsegrad_boost import PulseGradBoost
from experimental.pulsegrad_dual import PulseGradDual
from experimental.pulsegrad_poly import PulseGradPoly
from experimental.pulsegrad_h9 import PulseGradH9
from experimental.pulsegrad_h10 import PulseGradH10
from experimental.pulsegrad_h11 import PulseGradH11
from experimental.pulsegrad_h12 import PulseGradH12
from experimental.pulsegrad_h13 import PulseGradH13
from experimental.pulsegrad_h14 import PulseGradH14
from experimental.pulselion import PulseLion
from experimental.pulsesophia import PulseSophia
from experimental.pulselion_tuned import PulseLionStrong, PulseLionLinear, PulseLionAdaptive
from experimental.pulsesophia_tuned import PulseSophiaAdaptive
from experimental.pulseadam_adaptive import PulseAdamAdaptive
from experimental.pulsesgd_adaptive import PulseSGDAdaptive
from experimental.pulsediffgrad_adaptive import PulseDiffGradAdaptive
from experimental.pulseadadelta_adaptive import PulseAdaDeltaAdaptive
from experimental.pulsermsprop_adaptive import PulseRMSpropAdaptive
from experimental.pulseamsgrad_adaptive import PulseAMSGradAdaptive
from experimental.pulseadamw_adaptive import PulseAdamWAdaptive
from experimental.pulseadabelief_adaptive import PulseAdaBeliefAdaptive
from experimental.pulseadamp_adaptive import PulseAdamPAdaptive
from experimental.pulsemadgrad_adaptive import PulseMADGradAdaptive
from experimental.pulseadan_adaptive import PulseAdanAdaptive

def get_dataset(dataset_name, transform, root='./data'):
    """
    Return (train_set, test_set, num_classes) for the specified dataset_name.
    """
    dataset_name = dataset_name.lower()
    if dataset_name == 'cifar100':
        train_set = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform)
        test_set  = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform)
        num_classes = 100

    elif dataset_name == 'cifar10':
        train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
        test_set  = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
        num_classes = 10

    elif dataset_name == 'mnist':
        # MNIST and FMNIST are grayscale, so convert them to RGB
        train_set = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform)
        test_set  = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform)
        num_classes = 10

    elif dataset_name == 'fmnist':
        train_set = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
        test_set  = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
        num_classes = 10

    elif dataset_name == 'stl10':
        # STL10 has splits 'train' and 'test' for labeled data
        train_set = torchvision.datasets.STL10(root=root, split='train', download=True, transform=transform)
        test_set  = torchvision.datasets.STL10(root=root, split='test', download=True, transform=transform)
        num_classes = 10

    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    return train_set, test_set, num_classes

def get_model(model_name, num_classes):
    """
    Load a pretrained model and modify its final layer for num_classes.
    """
    model_name = model_name.lower()

    if model_name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
    elif model_name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == 'squeezenet1_0':
        model = torchvision.models.squeezenet1_0(pretrained=False)
        # SqueezeNet uses a Conv2d classifier; adjust its output channels
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1))
        model.num_classes = num_classes

    elif model_name == 'alexnet':
        model = torchvision.models.alexnet(pretrained=True)
        # The last linear layer is at index 6 of the classifier
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    elif model_name == 'vgg16':
        model = torchvision.models.vgg16(pretrained=True)
        # The last linear layer is at index 6 of the classifier
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    elif model_name == 'densenet121':
        model = torchvision.models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    elif model_name == 'googlenet':
        # Disable auxiliary classifiers for simplicity
        model = torchvision.models.googlenet(pretrained=True, aux_logits=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == 'shufflenet_v2_x1_0':
        model = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == 'mobilenet_v2':
        model = torchvision.models.mobilenet_v2(pretrained=True)
        # The final layer is classifier[1]
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == 'vit_b_16':
        model = torchvision.models.vit_b_16(pretrained=True)
        # Adjust the classification head for ViT
        if isinstance(model.heads, nn.Sequential):
            if len(model.heads) == 1:
                in_features = model.heads[0].in_features
                model.heads[0] = nn.Linear(in_features, num_classes)
            elif len(model.heads) > 1:
                in_features = model.heads[1].in_features
                model.heads[1] = nn.Linear(in_features, num_classes)
            else:
                raise ValueError("Unexpected structure of model.heads for vit_b_16.")
        else:
            # If model.heads is not a Sequential container, assume it's a single Linear layer
            in_features = model.heads.in_features
            model.heads = nn.Linear(in_features, num_classes)

    elif model_name == 'efficientnet_v2_s':
        model = torchvision.models.efficientnet_v2_s(pretrained=True)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unknown model {model_name}")

    return model

def get_optimizer(optimizer_name, model_params, lr):
    """
    Return an optimizer based on its name.
    """
    optimizer_name = optimizer_name.lower()
    if optimizer_name == 'adagrad':
        optimizer = optim.Adagrad(model_params, lr=lr)
    elif optimizer_name == 'adadelta':
        optimizer = optim.Adadelta(model_params, lr=lr)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model_params, lr=lr)
    elif optimizer_name == 'amsgrad':
        # AMSGrad is implemented as a flag in Adam
        optimizer = optim.Adam(model_params, lr=lr, amsgrad=True)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(model_params, lr=lr)
    elif optimizer_name == 'experimental':
        optimizer = Experimental(model_params, lr=lr)
    elif optimizer_name == 'experimental_logged':
        optimizer = ExperimentalLogged(model_params, lr=lr)
    elif optimizer_name == 'pulsegrad_enhanced':
        optimizer = PulseGradEnhanced(model_params, lr=lr)
    elif optimizer_name == 'pulsegrad_h2':
        optimizer = PulseGradH2(model_params, lr=lr)
    elif optimizer_name == 'pulsegrad_wd':
        optimizer = PulseGradWD(model_params, lr=lr)
    elif optimizer_name == 'pulsegrad_boost':
        optimizer = PulseGradBoost(model_params, lr=lr)
    elif optimizer_name == 'pulsegrad_dual':
        optimizer = PulseGradDual(model_params, lr=lr)
    elif optimizer_name == 'pulsegrad_poly':
        optimizer = PulseGradPoly(model_params, lr=lr)
    elif optimizer_name == 'pulsegrad_h9':
        optimizer = PulseGradH9(model_params, lr=lr)
    elif optimizer_name == 'pulsegrad_h10':
        optimizer = PulseGradH10(model_params, lr=lr)
    elif optimizer_name == 'pulsegrad_h11':
        optimizer = PulseGradH11(model_params, lr=lr)
    elif optimizer_name == 'pulsegrad_h12':
        optimizer = PulseGradH12(model_params, lr=lr, gamma_init=0.5, gamma_final=0.15, decay_rate=0.001)
    elif optimizer_name == 'pulsegrad_h13':
        optimizer = PulseGradH13(model_params, lr=lr, beta2_init=0.99, beta2_final=0.9999, decay_rate=0.001)
    elif optimizer_name == 'pulsegrad_h14':
        optimizer = PulseGradH14(model_params, lr=lr, gamma=0.0001)
    elif optimizer_name == 'experimentalv2':
        return ExperimentalV2(model_params, lr=lr, gamma=0.3)
    elif optimizer_name == 'diffgrad':
        optimizer = diffgrad(model_params, lr=lr)
    elif optimizer_name == 'adabelief':
        optimizer = AdaBelief(model_params, lr=lr, betas=(0.9, 0.999), eps=1e-8)

    elif optimizer_name == 'adamp':
        optimizer = AdamP(model_params, lr=lr, betas=(0.9, 0.999), weight_decay=1e-2)

    elif optimizer_name == 'madgrad':
        optimizer = madgrad.MADGRAD(model_params, lr=lr, momentum=0.9, weight_decay=1e-6)

    elif optimizer_name == 'adan':
        # three betas per the paper: β1, β2, β3
        optimizer = Adan(model_params, lr=lr, betas=(0.98, 0.92, 0.99),
                         weight_decay=1e-2)

    elif optimizer_name == 'lion':
        optimizer = Lion(model_params, lr=lr, weight_decay=1e-2)

    elif optimizer_name == 'pulselion':
        optimizer = PulseLion(model_params, lr=lr, gamma=0.3, weight_decay=1e-2)

    elif optimizer_name == 'pulselion_strong':
        optimizer = PulseLionStrong(model_params, lr=lr, gamma=10.0, weight_decay=1e-2)

    elif optimizer_name == 'pulselion_linear':
        optimizer = PulseLionLinear(model_params, lr=lr, gamma=1.0, weight_decay=1e-2)

    elif optimizer_name == 'pulselion_adaptive':
        optimizer = PulseLionAdaptive(model_params, lr=lr, gamma=0.5, weight_decay=1e-2)

    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model_params, lr=lr)

    elif optimizer_name == 'sgd_momentum':
        optimizer = optim.SGD(model_params, lr=lr, momentum=0.9)

    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model_params, lr=lr, weight_decay=1e-2)

    elif optimizer_name == 'sophia':
        # Sophia-G default hyper-params from authors
        optimizer = SophiaG(model_params, lr=lr, betas=(0.965, 0.99),
                            rho=0.04, weight_decay=1e-1)

    elif optimizer_name == 'pulsesophia':
        optimizer = PulseSophia(model_params, lr=lr, betas=(0.965, 0.99),
                                rho=0.04, gamma=0.3, weight_decay=1e-1)

    elif optimizer_name == 'pulsesophia_adaptive':
        optimizer = PulseSophiaAdaptive(model_params, lr=lr, betas=(0.965, 0.99),
                                        rho=0.04, gamma=0.5, weight_decay=1e-1)

    elif optimizer_name == 'pulseadam_adaptive':
        optimizer = PulseAdamAdaptive(model_params, lr=lr, betas=(0.9, 0.999),
                                     gamma=0.5, weight_decay=1e-2)

    elif optimizer_name == 'pulsesgd_adaptive':
        optimizer = PulseSGDAdaptive(model_params, lr=lr, momentum=0.9,
                                    gamma=0.5, weight_decay=1e-2)

    elif optimizer_name == 'pulsediffgrad_adaptive':
        optimizer = PulseDiffGradAdaptive(model_params, lr=lr, betas=(0.9, 0.999),
                                         gamma=0.5, weight_decay=1e-2)

    elif optimizer_name == 'pulseadadelta_adaptive':
        optimizer = PulseAdaDeltaAdaptive(model_params, lr=lr, rho=0.9,
                                          gamma=0.5, weight_decay=1e-2)

    elif optimizer_name == 'pulsermsprop_adaptive':
        optimizer = PulseRMSpropAdaptive(model_params, lr=lr, alpha=0.99, momentum=0,
                                         gamma=0.5, weight_decay=1e-2)

    elif optimizer_name == 'pulseamsgrad_adaptive':
        optimizer = PulseAMSGradAdaptive(model_params, lr=lr, betas=(0.9, 0.999),
                                         gamma=0.5, weight_decay=1e-2)

    elif optimizer_name == 'pulseadamw_adaptive':
        optimizer = PulseAdamWAdaptive(model_params, lr=lr, betas=(0.9, 0.999),
                                       gamma=0.5, weight_decay=1e-2)

    elif optimizer_name == 'pulseadabelief_adaptive':
        optimizer = PulseAdaBeliefAdaptive(model_params, lr=lr, betas=(0.9, 0.999),
                                           gamma=0.5, weight_decay=1e-2)

    elif optimizer_name == 'pulseadamp_adaptive':
        optimizer = PulseAdamPAdaptive(model_params, lr=lr, betas=(0.9, 0.999),
                                       gamma=0.5, weight_decay=1e-2)

    elif optimizer_name == 'pulsemadgrad_adaptive':
        optimizer = PulseMADGradAdaptive(model_params, lr=lr, momentum=0.9,
                                         gamma=0.5, weight_decay=1e-2)

    elif optimizer_name == 'pulseadan_adaptive':
        optimizer = PulseAdanAdaptive(model_params, lr=lr, betas=(0.98, 0.92, 0.99),
                                      gamma=0.5, weight_decay=1e-2)

    else:
        raise ValueError(f"Unknown optimizer {optimizer_name}")

    return optimizer

# ==============================================================================
# Learning Rate Search Functions
# ==============================================================================

def get_config_key(dataset, model, optimizer, batch_size, epochs):
    """
    Generate a unique key for the configuration (excluding lr and k_folds).
    """
    config_str = f"{dataset}_{model}_{optimizer}_{batch_size}_{epochs}"
    return hashlib.md5(config_str.encode()).hexdigest()

def load_lr_cache(cache_file):
    """
    Load the LR search cache from JSON file.
    Returns a dictionary mapping config keys to best learning rates.
    """
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_lr_cache(cache, cache_file):
    """
    Save the LR search cache to JSON file.
    """
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(cache, f, indent=2)

def perform_lr_search(dataset_name, model_name, optimizer_name, batch_size,
                     epochs_per_lr, full_train_set, transform, num_classes, device, seed):
    """
    Perform learning rate search by testing 15 learning rates from 1e-4 to 1e-1.
    Each LR is tested for 10 epochs on a single train/val split (no k-fold).
    Returns the best learning rate based on validation accuracy.
    """
    print("\n" + "="*80)
    print("LEARNING RATE SEARCH")
    print("="*80)
    print(f"Testing 15 learning rates from 1e-4 to 1e-1")
    print(f"Each LR trained for {epochs_per_lr} epochs on a single train/val split")
    print("="*80 + "\n")

    # Generate 15 learning rates with uniform spacing within each order of magnitude
    # 5 LRs between 1e-4 and 1e-3, 5 between 1e-3 and 1e-2, 5 between 1e-2 and 1e-1
    lr_range_1 = np.linspace(1e-4, 1e-3, 5, endpoint=False)  # 1e-4 to 1e-3
    lr_range_2 = np.linspace(1e-3, 1e-2, 5, endpoint=False)  # 1e-3 to 1e-2
    lr_range_3 = np.linspace(1e-2, 1e-1, 5, endpoint=False)  # 1e-2 to 1e-1
    lr_candidates = np.concatenate([lr_range_1, lr_range_2, lr_range_3])

    # Create a single train/val split (80/20)
    dataset_size = len(full_train_set)
    indices = list(range(dataset_size))
    split_idx = int(0.8 * dataset_size)

    # Shuffle with fixed seed for reproducibility
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    train_subset = Subset(full_train_set, train_idx)
    val_subset = Subset(full_train_set, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    best_lr = None
    best_val_acc = 0.0
    lr_results = []

    for i, lr in enumerate(lr_candidates, 1):
        print(f"\n[{i}/15] Testing LR = {lr:.6f}")
        print("-" * 40)

        # Create fresh model and optimizer for this LR
        model = get_model(model_name, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = get_optimizer(optimizer_name, model.parameters(), lr)

        # Train for epochs_per_lr epochs
        best_acc_for_this_lr = 0.0

        for epoch in range(1, epochs_per_lr + 1):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

            train_loss = train_loss / train_total
            train_acc = 100.0 * train_correct / train_total

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_loss = val_loss / val_total
            val_acc = 100.0 * val_correct / val_total

            if val_acc > best_acc_for_this_lr:
                best_acc_for_this_lr = val_acc

            # Print progress for first, middle, and last epochs
            if epoch == 1 or epoch == epochs_per_lr // 2 or epoch == epochs_per_lr:
                print(f"  Epoch [{epoch}/{epochs_per_lr}] Val Acc: {val_acc:.2f}%")

        print(f"  Best Val Acc: {best_acc_for_this_lr:.2f}%")

        lr_results.append({
            'lr': float(lr),
            'best_val_acc': best_acc_for_this_lr
        })

        # Update best LR if this one is better
        if best_acc_for_this_lr > best_val_acc:
            best_val_acc = best_acc_for_this_lr
            best_lr = lr

    print("\n" + "="*80)
    print("LEARNING RATE SEARCH COMPLETE")
    print("="*80)
    print(f"Best LR: {best_lr:.6f} (Val Acc: {best_val_acc:.2f}%)")
    print("="*80 + "\n")

    # Return best LR and all results for caching
    return best_lr, lr_results

def train_one_fold(model, train_loader, val_loader, criterion, optimizer, device, epochs):
    """
    Train the model for 'epochs' on the given train_loader, validate on val_loader.
    Returns:
      best_acc, best_f1, best_acc_epoch, best_f1_epoch
    where those metrics are the best (highest) observed during training.
    """
    best_acc = 0.0
    best_f1 = 0.0
    best_acc_epoch = 0
    best_f1_epoch = 0

    for epoch in range(1, epochs + 1):

        if epoch == int(0.8 * epochs) + 1:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1
        # -------------------- TRAIN --------------------
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        train_acc = 100.0 * correct_train / total_train
        train_loss = running_loss / total_train

        # -------------------- VALIDATE --------------------
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = 100.0 * correct_val / total_val
        val_loss = running_val_loss / total_val
        val_f1 = f1_score(all_labels, all_preds, average='macro')

        # Track best accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            best_acc_epoch = epoch

        # Track best F1
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_f1_epoch = epoch
        

        print(f"Epoch [{epoch}/{epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}")

    return best_acc, best_f1, best_acc_epoch, best_f1_epoch

def main():
    parser = argparse.ArgumentParser(description="General Training Script with K-Fold Cross Validation")
    parser.add_argument('--dataset', type=str, default='CIFAR100',
                        choices=['CIFAR100', 'CIFAR10', 'MNIST', 'FMNIST', 'STL10'],
                        help="Dataset to use")
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=[
                            'resnet18',
                            'resnet50',
                            'squeezenet1_0',
                            'alexnet',
                            'vgg16',
                            'densenet121',
                            'googlenet',
                            'shufflenet_v2_x1_0',
                            'mobilenet_v2',
                            'vit_b_16',
                            'efficientnet_v2_s'
                        ],
                        help="Model architecture to use")
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adagrad', 'adadelta', 'rmsprop', 'amsgrad', 'adam', 'experimental', 'experimental_logged', 'pulsegrad_enhanced', 'pulsegrad_h2', 'pulsegrad_wd', 'pulsegrad_boost', 'pulsegrad_dual', 'pulsegrad_poly', 'pulsegrad_h9', 'pulsegrad_h10', 'pulsegrad_h11', 'pulsegrad_h12', 'pulsegrad_h13', 'pulsegrad_h14', 'diffgrad','adabelief', 'adamp', 'madgrad', 'adan', 'lion', 'pulselion', 'pulselion_strong', 'pulselion_linear', 'pulselion_adaptive', 'adahessian', 'sophia', 'pulsesophia', 'pulsesophia_adaptive', 'pulseadam_adaptive', 'pulsesgd_adaptive', 'pulsediffgrad_adaptive', 'pulseadadelta_adaptive', 'pulsermsprop_adaptive', 'pulseamsgrad_adaptive', 'pulseadamw_adaptive', 'pulseadabelief_adaptive', 'pulseadamp_adaptive', 'pulsemadgrad_adaptive', 'pulseadan_adaptive', 'experimentalv2', 'sgd', 'sgd_momentum', 'adamw'],
                        help="Optimizer to use")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--k_folds', type=int, default=5, help="Number of folds for K-Fold cross validation")
    parser.add_argument('--lr_study', action='store_true',
                        help="Perform learning rate search (15 LRs from 1e-4 to 1e-1, 10 epochs each)")
    args = parser.parse_args()

    # ------------------------------
    # Set seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ------------------------------
    # Prepare transformations
    # For pretrained models we use 224x224 and ImageNet normalization.
    # MNIST, FMNIST are grayscale so convert them to RGB.
    base_transforms = [transforms.Resize(224)]
    if args.dataset.lower() in ['mnist', 'fmnist']:
        base_transforms.append(transforms.Lambda(lambda image: image.convert("RGB")))
    base_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    transform = transforms.Compose(base_transforms)

    # ------------------------------
    # Load dataset and test set (we won't use the test set in cross-validation,
    # but we keep it loaded for completeness)
    full_train_set, test_set, num_classes = get_dataset(args.dataset, transform)

    # ------------------------------
    # Learning Rate Search (if requested)
    used_lr = args.lr  # Default to user-provided LR

    if args.lr_study:
        lr_cache_file = "./results/lr_search_cache.json"
        lr_cache = load_lr_cache(lr_cache_file)

        # Generate config key for this experiment
        config_key = get_config_key(args.dataset, args.model, args.optimizer,
                                    args.batch_size, args.epochs)

        # Check if LR for this config is already cached
        if config_key in lr_cache:
            used_lr = lr_cache[config_key]['best_lr']
            print("\n" + "="*80)
            print("USING CACHED LEARNING RATE")
            print("="*80)
            print(f"Configuration: {args.dataset}, {args.model}, {args.optimizer}")
            print(f"Batch Size: {args.batch_size}, Epochs: {args.epochs}")
            print(f"Cached LR: {used_lr:.6f}")
            print("="*80 + "\n")
        else:
            # Perform LR search
            print(f"\nNo cached LR found for this configuration. Performing LR search...")
            best_lr, lr_results = perform_lr_search(
                args.dataset, args.model, args.optimizer, args.batch_size,
                epochs_per_lr=10, full_train_set=full_train_set,
                transform=transform, num_classes=num_classes,
                device=device, seed=seed
            )
            used_lr = best_lr

            # Cache the result
            lr_cache[config_key] = {
                'dataset': args.dataset,
                'model': args.model,
                'optimizer': args.optimizer,
                'batch_size': args.batch_size,
                'epochs': args.epochs,
                'best_lr': float(best_lr),
                'lr_results': lr_results
            }
            save_lr_cache(lr_cache, lr_cache_file)
            print(f"LR search results saved to {lr_cache_file}\n")

    # Use the determined learning rate (either from search or user-provided)
    print(f"Using Learning Rate: {used_lr:.6f}\n")

    # ------------------------------
    # Setup K-Fold cross validation
    kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=seed)

    # Prepare lists to collect best scores across folds
    fold_best_accs = []
    fold_best_f1s = []
    fold_best_acc_epochs = []
    fold_best_f1_epochs = []

    print(f"\nStarting {args.k_folds}-Fold Cross Validation for dataset={args.dataset}...\n")

    # Create indices for the entire training set
    dataset_indices = list(range(len(full_train_set)))

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset_indices), start=1):
        print(f"\n--- Fold {fold} / {args.k_folds} ---")
        # Subset the dataset for this fold
        train_subset = Subset(full_train_set, train_idx)
        val_subset = Subset(full_train_set, val_idx)

        # Create data loaders
        train_loader = DataLoader(train_subset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size,
                                shuffle=False, num_workers=0)

        # ------------------------------
        # Load a fresh model for each fold
        model = get_model(args.model, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = get_optimizer(args.optimizer, model.parameters(), used_lr)

        # Train for this fold
        best_acc, best_f1, best_acc_epoch, best_f1_epoch = train_one_fold(
            model, train_loader, val_loader, criterion, optimizer, device, args.epochs
        )

        # Record the best results of this fold
        fold_best_accs.append(best_acc)
        fold_best_f1s.append(best_f1)
        fold_best_acc_epochs.append(best_acc_epoch)
        fold_best_f1_epochs.append(best_f1_epoch)

    # ------------------------------
    # Compute mean/std of best metrics across all folds
    mean_acc = np.mean(fold_best_accs)
    std_acc = np.std(fold_best_accs)

    mean_f1 = np.mean(fold_best_f1s)
    std_f1 = np.std(fold_best_f1s)

    mean_acc_epoch = np.mean(fold_best_acc_epochs)
    std_acc_epoch = np.std(fold_best_acc_epochs)

    mean_f1_epoch = np.mean(fold_best_f1_epochs)
    std_f1_epoch = np.std(fold_best_f1_epochs)

    print("\nCross Validation Results:")
    print(f"Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"F1-score: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"Best Accuracy Epoch: {mean_acc_epoch:.2f} ± {std_acc_epoch:.2f}")
    print(f"Best F1 Epoch: {mean_f1_epoch:.2f} ± {std_f1_epoch:.2f}")

    # ------------------------------
    # Write results to CSV
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    csv_file = os.path.join(results_dir, "classification_results.csv")

    # Define CSV header and row
    header = [
        "timestamp", "dataset", "model", "optimizer", "epochs", "batch_size", "lr", "k_folds",
        "best_val_acc_mean", "best_val_acc_std",
        "best_val_f1_mean", "best_val_f1_std",
        "best_acc_epoch_mean", "best_acc_epoch_std",
        "best_f1_epoch_mean", "best_f1_epoch_std"
    ]

    row = [
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        args.dataset,
        args.model,
        args.optimizer,
        args.epochs,
        args.batch_size,
        used_lr,
        args.k_folds,
        f"{mean_acc:.4f}",
        f"{std_acc:.4f}",
        f"{mean_f1:.4f}",
        f"{std_f1:.4f}",
        f"{mean_acc_epoch:.2f}",
        f"{std_acc_epoch:.2f}",
        f"{mean_f1_epoch:.2f}",
        f"{std_f1_epoch:.2f}",
    ]

    file_exists = os.path.isfile(csv_file)
    rows = []

    # If file exists, read it in and filter out any row that has exactly the same combo of
    # [dataset, model, optimizer, epochs, batch_size, lr, k_folds]
    if file_exists:
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            existing_header = next(reader, None)  # Read header row
            for existing_row in reader:
                if (existing_row[1] == args.dataset and
                    existing_row[2] == args.model and
                    existing_row[3] == args.optimizer and
                    existing_row[4] == str(args.epochs) and
                    existing_row[5] == str(args.batch_size) and
                    existing_row[6] == str(args.lr) and
                    existing_row[7] == str(args.k_folds)):
                    # Skip this row to overwrite
                    continue
                rows.append(existing_row)

    # Append the newly computed row
    rows.append(row)

    # Write everything back, including the header
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\nResults saved (overwritten if matching combo) to {csv_file}")

if __name__ == '__main__':
    main()

