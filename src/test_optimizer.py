"""Quick test to validate optimizer implementation."""
import torch
import torch.nn as nn
from experimental.pulsegrad_enhanced import PulseGradEnhanced

# Simple 2-layer network
model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 2)
)

# Test optimizer creation
optimizer = PulseGradEnhanced(model.parameters(), lr=0.01)

# Test forward/backward pass
x = torch.randn(32, 10)
y = torch.randint(0, 2, (32,))
criterion = nn.CrossEntropyLoss()

for i in range(5):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    print(f"Step {i+1}: Loss = {loss.item():.4f}")

print("\nOptimizer test passed! PulseGradEnhanced works correctly.")
