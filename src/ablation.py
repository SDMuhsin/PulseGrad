#!/usr/bin/env python3
"""Ablation study script for DiffGrad vs. PulseGrad (Experimental).

This version prints detailed progress so you can follow training in real time.

Usage examples
--------------
python ablation_diffgrad_pulsegrad.py --ablation lr --plot
python ablation_diffgrad_pulsegrad.py --ablation lr --plot --skip_run  # just plotting
"""
import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

# ---------------------------------------------------------------------------
# Local optimizers
# ---------------------------------------------------------------------------
from experimental.diffgrad import diffgrad  # type: ignore
from experimental.exp import Experimental as PulseGrad  # rename for clarity


# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Dataset / model helpers
# ---------------------------------------------------------------------------

def get_dataset(name: str, transform, root: str = "./data"):
    name = name.lower()
    if name == "cifar100":
        train = torchvision.datasets.CIFAR100(root, train=True, download=True, transform=transform)
        test  = torchvision.datasets.CIFAR100(root, train=False,  download=True, transform=transform)
        return train, test, 100
    if name == "cifar10":
        train = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=transform)
        test  = torchvision.datasets.CIFAR10(root, train=False,  download=True, transform=transform)
        return train, test, 10
    raise ValueError(name)


def get_model(name: str, num_classes: int):
    name = name.lower()
    if name == "resnet18":
        model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(name)


def build_optimizer(opt_name: str, params, lr: float, betas=(0.9, 0.999), eps: float = 1e-8, gamma: float = 0.3):
    opt_name = opt_name.lower()
    if opt_name == "diffgrad":
        return diffgrad(params, lr=lr, betas=betas, eps=eps)
    if opt_name == "pulsegrad":
        return PulseGrad(params, lr=lr, betas=betas, gamma=gamma, eps=eps)
    raise ValueError(opt_name)


# ---------------------------------------------------------------------------
# Training utilities (with verbose prints)
# ---------------------------------------------------------------------------

def train_one_fold(model, train_loader, val_loader, criterion, optimizer, device, epochs):
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        if False and epoch == int(0.8 * epochs) + 1: # Interferes with ablation so cancelled
            for g in optimizer.param_groups:
                g["lr"] *= 0.1
                print(f"    LR decayed to {g['lr']:.2e}")

        # ----------------- train -----------------
        model.train()
        correct, total = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total   += y.size(0)
        train_acc = 100.0 * correct / total

        # ----------------- validate --------------
        model.eval(); correct_val, total_val = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                pred = out.argmax(1)
                correct_val += (pred == y).sum().item()
                total_val   += y.size(0)
        val_acc = 100.0 * correct_val / total_val
        best_acc = max(best_acc, val_acc)

        print(f"      Epoch {epoch:02d}/{epochs}: train_acc={train_acc:5.2f}% | val_acc={val_acc:5.2f}% | best={best_acc:5.2f}%", flush=True)
    return best_acc


def run_cv(dataset_name: str, model_name: str, optimizer_name: str, param_cfg: dict,
           k_folds: int, epochs: int, batch_size: int, seed: int = 42):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    full_train, _, n_cls = get_dataset(dataset_name, transform)
    indices = list(range(len(full_train)))
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

    scores = []
    for fold, (tr_idx, val_idx) in enumerate(kfold.split(indices), 1):
        print(f"    Fold {fold}/{k_folds} — {optimizer_name} — params: {param_cfg}")
        tr_set, val_set = Subset(full_train, tr_idx), Subset(full_train, val_idx)
        tr_loader = DataLoader(tr_set, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

        model = get_model(model_name, n_cls).to(device)
        optim = build_optimizer(optimizer_name, model.parameters(), **param_cfg)
        acc = train_one_fold(model, tr_loader, val_loader, nn.CrossEntropyLoss(), optim, device, epochs)
        print(f"    >>> Fold {fold} finished — best_val_acc={acc:.2f}%\n")
        scores.append(acc)
    mean, std = float(np.mean(scores)), float(np.std(scores))
    print(f"    === {optimizer_name} done: {mean:.2f} ± {std:.2f}% ===\n")
    return mean, std


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_ablation(param_name: str, values, diff_stats, pulse_stats, out_file: Path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    dg_mean, dg_std = zip(*diff_stats)
    pg_mean, pg_std = zip(*pulse_stats)
    plt.errorbar(values, dg_mean, yerr=dg_std, marker="o", label="DiffGrad", capsize=3)
    plt.errorbar(values, pg_mean, yerr=pg_std, marker="s", label="PulseGrad", capsize=3)
    label_map = {
        "lr": "Learning rate (log10)",
        "betas": "$\\beta_1$",  # first‑moment
        "gamma": "$\\gamma$",   # PulseGrad hyper‑parameter
        "eps": "$\\epsilon$",
    }
    plt.xlabel(label_map[param_name])
    plt.ylabel("Top‑1 Accuracy (%)")
    plt.title(f"Ablation on {label_map[param_name]}")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Plot saved to {out_file}")


# ---------------------------------------------------------------------------
# Command‑line interface
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ablation study: DiffGrad vs PulseGrad (with verbose output)")
    parser.add_argument("--ablation", required=True, choices=["lr", "betas", "gamma", "eps"], help="Hyper‑parameter to vary")
    parser.add_argument("--dataset", default="CIFAR10")
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--k_folds", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--plot", action="store_true", help="Plot results after run (or only plot if --skip_run)")
    parser.add_argument("--skip_run", action="store_true", help="Skip running ablation if JSON exists")
    args = parser.parse_args()

    sweep = {
        "lr":    [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
        "betas": [0.5, 0.7, 0.9, 0.95],
        "gamma": [0.1, 0.3, 0.5, 0.7],
        "eps":   [1e-8, 1e-7, 1e-6, 1e-5, 1e-4],
    }
    values = sweep[args.ablation]

    results_dir = Path("./results"); results_dir.mkdir(exist_ok=True)
    json_path = results_dir / f"ablation_{args.ablation}.json"

    if args.skip_run and json_path.exists():
        print("Loading cached results …")
        data = json.loads(json_path.read_text())
        values = data["param_values"]
        diff_stats = list(zip(data["DiffGrad_mean"], data["DiffGrad_std"]))
        pulse_stats = list(zip(data["PulseGrad_mean"], data["PulseGrad_std"]))
    else:
        diff_stats, pulse_stats = [], []
        for val in values:
            print(f"\n>>> Running ablation {args.ablation} = {val}")
            common_cfg = {"lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8, "gamma": 0.3}
            if args.ablation == "lr":
                common_cfg["lr"] = val
            elif args.ablation == "betas":
                common_cfg["betas"] = (val, 0.999)
            elif args.ablation == "gamma":
                common_cfg["gamma"] = val
            elif args.ablation == "eps":
                common_cfg["eps"] = val

            print("  -- DiffGrad --")
            mean_dg, std_dg = run_cv(args.dataset, args.model, "diffgrad", common_cfg,
                                     args.k_folds, args.epochs, args.batch_size)
            diff_stats.append((mean_dg, std_dg))

            print("  -- PulseGrad --")
            mean_pg, std_pg = run_cv(args.dataset, args.model, "pulsegrad", common_cfg,
                                     args.k_folds, args.epochs, args.batch_size)
            pulse_stats.append((mean_pg, std_pg))

        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "ablation": args.ablation,
            "param_values": values,
            "DiffGrad_mean": [m for m, _ in diff_stats],
            "DiffGrad_std":  [s for _, s in diff_stats],
            "PulseGrad_mean": [m for m, _ in pulse_stats],
            "PulseGrad_std":  [s for _, s in pulse_stats],
            "meta": {
                "dataset": args.dataset,
                "model": args.model,
                "epochs": args.epochs,
                "k_folds": args.k_folds,
            },
        }
        json_path.write_text(json.dumps(payload, indent=2))
        print(f"Results saved to {json_path}")

    if args.plot:
        plot_ablation(args.ablation, values, diff_stats, pulse_stats,
                      results_dir / f"ablation_{args.ablation}.png")


if __name__ == "__main__":
    main()

