"""
Diagnostic script to analyze pulse optimizer behavior.

Runs PulseLion and PulseSophia on CIFAR100 for a few epochs,
logging key metrics to understand damping characteristics.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experimental.pulselion_diagnostic import PulseLionDiagnostic
from experimental.pulsesophia_diagnostic import PulseSophiaDiagnostic


def get_model():
    """Get ResNet18 for CIFAR100"""
    from torchvision.models import resnet18
    model = resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 100)
    return model


def get_data(batch_size=64, subset_size=10000):
    """Get CIFAR100 data (subset for speed)"""
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    trainset = torchvision.datasets.CIFAR100(root='./cache', train=True, download=True, transform=transform)

    # Use subset for faster diagnostics
    indices = list(range(subset_size))
    trainset = Subset(trainset, indices)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    return train_loader


def run_diagnostic(optimizer_name, optimizer, train_loader, device, epochs=3):
    """Run training with diagnostics"""
    print(f"\n{'='*60}")
    print(f"Running {optimizer_name} diagnostics...")
    print(f"{'='*60}\n")

    model = get_model().to(device)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs} [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {running_loss/(batch_idx+1):.3f} | "
                      f"Acc: {100.*correct/total:.2f}%")

        print(f"\nEpoch {epoch+1} completed: Loss: {running_loss/len(train_loader):.3f} | Acc: {100.*correct/total:.2f}%\n")


def main():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Get data
    train_loader = get_data()

    # Test different learning rates and gammas
    configs = [
        # PulseLion with different gammas
        ('PulseLion (LR=0.001, gamma=0.3)', PulseLionDiagnostic,
         {'lr': 0.001, 'gamma': 0.3, 'weight_decay': 1e-2,
          'diagnostic_file': 'results/diagnostics/pulselion_lr0001_g03.csv'}),
        ('PulseLion (LR=0.001, gamma=0.1)', PulseLionDiagnostic,
         {'lr': 0.001, 'gamma': 0.1, 'weight_decay': 1e-2,
          'diagnostic_file': 'results/diagnostics/pulselion_lr0001_g01.csv'}),
        ('PulseLion (LR=0.0015, gamma=0.3)', PulseLionDiagnostic,
         {'lr': 0.0015, 'gamma': 0.3, 'weight_decay': 1e-2,
          'diagnostic_file': 'results/diagnostics/pulselion_lr0015_g03.csv'}),

        # PulseSophia with different gammas
        ('PulseSophia (LR=0.001, gamma=0.3)', PulseSophiaDiagnostic,
         {'lr': 0.001, 'gamma': 0.3, 'weight_decay': 1e-1, 'rho': 0.04,
          'diagnostic_file': 'results/diagnostics/pulsesophia_lr0001_g03.csv'}),
        ('PulseSophia (LR=0.001, gamma=0.1)', PulseSophiaDiagnostic,
         {'lr': 0.001, 'gamma': 0.1, 'weight_decay': 1e-1, 'rho': 0.04,
          'diagnostic_file': 'results/diagnostics/pulsesophia_lr0001_g01.csv'}),
        ('PulseSophia (LR=0.0015, gamma=0.3)', PulseSophiaDiagnostic,
         {'lr': 0.0015, 'gamma': 0.3, 'weight_decay': 1e-1, 'rho': 0.04,
          'diagnostic_file': 'results/diagnostics/pulsesophia_lr0015_g03.csv'}),
    ]

    os.makedirs('results/diagnostics', exist_ok=True)

    for name, optimizer_class, kwargs in configs:
        model = get_model().to(device)
        optimizer = optimizer_class(model.parameters(), **kwargs)
        run_diagnostic(name, optimizer, train_loader, device, epochs=3)

    print("\n" + "="*60)
    print("Diagnostics complete! Check results/diagnostics/*.csv")
    print("="*60)


if __name__ == '__main__':
    main()
