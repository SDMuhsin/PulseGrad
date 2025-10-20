"""Quick test of diagnostic logging"""
import torch
import torch.nn as nn
from experimental.pulselion_diagnostic import PulseLionDiagnostic
import os

# Simple model
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 2)
).cuda(1)

# Create optimizer
optimizer = PulseLionDiagnostic(
    model.parameters(),
    lr=0.001,
    gamma=0.3,
    diagnostic_file='results/diagnostics/test.csv'
)

print("Running quick test...")
for step in range(5):
    # Fake data
    x = torch.randn(32, 10).cuda(1)
    y = torch.randint(0, 2, (32,)).cuda(1)

    # Forward
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, y)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Step {step}: loss={loss.item():.4f}")

print("\nChecking file...")
if os.path.exists('results/diagnostics/test.csv'):
    with open('results/diagnostics/test.csv') as f:
        print(f.read())
else:
    print("File not created!")
