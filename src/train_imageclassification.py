import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import os
import csv
from datetime import datetime

# Local experimental optimizers (ensure these modules are in your PYTHONPATH)
from experimental.lance import LANCE
from experimental.diffgrad import diffgrad
from experimental.cosmic import COSMIC
from experimental.exp import Experimental

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
import os
import ast
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
from sklearn.model_selection import StratifiedKFold
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error

# Define device and ensure using cuda:1
device = torch.device("cuda:1")

# Ensure results directory exists
os.makedirs("./results", exist_ok=True)

# 1. Read + Prepare Train Data
train = pd.read_csv("./data/Train.csv")

# Create placement and img_origin mappers just like the original code
placement_mapper = train[["ID", "placement"]].drop_duplicates().set_index("ID").to_dict()
img_origin_mapper = train[["ID", "img_origin"]].drop_duplicates().set_index("ID").to_dict()

# -- We now aggregate polygons by ID so we can draw them all on one mask --
# Group polygons into lists under each ID
poly_agg = train.groupby("ID")["polygon"].agg(list).reset_index(name="polygons")

# Aggregate by ID to get sums for boil_nbr and pan_nbr
train_df = train.groupby("ID", as_index=False).sum()[["ID", "boil_nbr", "pan_nbr"]]

# Map back img_origin and placement
train_df["img_origin"] = train_df["ID"].map(img_origin_mapper["img_origin"])
train_df["placement"] = train_df["ID"].map(placement_mapper["placement"])
train_df["path"] = "./data/images/" + train_df["ID"] + ".jpg"

# Merge the aggregated polygon lists so each ID has a single polygons field
train_df = train_df.merge(poly_agg, on="ID", how="left")

# For StratifiedKFold (like original code)
train_df["stratify_label"] = train_df[["boil_nbr", "pan_nbr"]].sum(axis=1)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_df["fold"] = -1
for f, (_, val_idx) in enumerate(skf.split(train_df, train_df["stratify_label"])):
    train_df.loc[val_idx, "fold"] = f
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == 'squeezenet1_0':
        model = torchvision.models.squeezenet1_0(pretrained=True)
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

    elif model_name == 'densenet161':
        model = torchvision.models.densenet161(pretrained=True)
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
    elif optimizer_name == 'diffgrad':
        optimizer = diffgrad(model_params, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer {optimizer_name}")

    return optimizer

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
                            'densenet161',
                            'googlenet',
                            'shufflenet_v2_x1_0',
                            'mobilenet_v2',
                            'vit_b_16',
                            'efficientnet_v2_s'
                        ],
                        help="Model architecture to use")
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adagrad', 'adadelta', 'rmsprop', 'amsgrad', 'adam', 'experimental', 'diffgrad'],
                        help="Optimizer to use")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--k_folds', type=int, default=5, help="Number of folds for K-Fold cross validation")
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
                                  shuffle=True, num_workers=2)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size,
                                shuffle=False, num_workers=2)

        # ------------------------------
        # Load a fresh model for each fold
        model = get_model(args.model, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = get_optimizer(args.optimizer, model.parameters(), args.lr)

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
        args.lr,
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

