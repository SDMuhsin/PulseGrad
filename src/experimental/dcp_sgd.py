"""
Directional Consistency Pulse SGD (DCP-SGD)

Theoretical Foundation:
-----------------------
SGD's strength lies in its stochastic noise for exploration, but suffers from
directional instability. This optimizer applies stabilization ONLY when gradient
directions are consistent, and removes it when directions oppose.

Key Insight:
- Consistent directions (cos_sim > 0) → Apply pulse to accelerate convergence
- Opposing directions (cos_sim < 0) → Remove pulse to preserve exploration
- No normalization by gradient magnitude → Avoids premature convergence

Mathematical Formulation:
-------------------------
cos_similarity = (g_t · g_{t-1}) / (||g_t|| ||g_{t-1}|| + ε)
consistency_weight = max(0, cos_similarity)^α  # α controls emphasis
pulse = consistency_weight * γ * ||g_{t-1}||
stabilizer = 1 + pulse / (||g_t|| + ε)
g_stabilized = g_t / stabilizer

Why This Works:
---------------
1. Zero pulse when directions oppose → Full exploration in rough regions
2. Maximum pulse when aligned → Accelerated descent in smooth regions
3. Pulse strength proportional to ||g_{t-1}|| → Natural scaling
4. No gradient magnitude normalization → No premature convergence
5. Exploits SGD's directional statistics, not magnitude statistics
"""

import torch
from torch.optim.optimizer import Optimizer

class DCPSGD(Optimizer):
    """
    Directional Consistency Pulse SGD

    Args:
        params (iterable): Iterable of parameters to optimize
        lr (float): Learning rate
        gamma (float, optional): Pulse strength coefficient (default: 0.1)
        alpha (float, optional): Consistency emphasis power (default: 2.0)
        momentum (float, optional): Momentum factor (default: 0)
        dampening (float, optional): Dampening for momentum (default: 0)
        weight_decay (float, optional): Weight decay (L2 penalty) (default: 0)
        nesterov (bool, optional): Enable Nesterov momentum (default: False)
    """

    def __init__(self, params, lr, gamma=0.1, alpha=2.0, momentum=0,
                 dampening=0, weight_decay=0, nesterov=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if gamma < 0.0:
            raise ValueError(f"Invalid gamma value: {gamma}")
        if alpha < 0.0:
            raise ValueError(f"Invalid alpha value: {alpha}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, gamma=gamma, alpha=alpha, momentum=momentum,
                       dampening=dampening, weight_decay=weight_decay,
                       nesterov=nesterov)

        super(DCPSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(DCPSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                                         and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            gamma = group['gamma']
            alpha = group['alpha']
            momentum = group['momentum']
            dampening = group['dampening']
            weight_decay = group['weight_decay']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # Apply weight decay (before any modifications)
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                param_state = self.state[p]

                # Initialize state on first step
                if len(param_state) == 0:
                    param_state['step'] = 0
                    param_state['prev_grad'] = torch.zeros_like(grad)
                    if momentum != 0:
                        param_state['momentum_buffer'] = torch.zeros_like(grad)

                param_state['step'] += 1
                prev_grad = param_state['prev_grad']

                # Compute directional consistency pulse (only after first step)
                if param_state['step'] > 1:
                    # Compute cosine similarity between current and previous gradients
                    grad_norm = torch.norm(grad)
                    prev_grad_norm = torch.norm(prev_grad)

                    eps = 1e-8

                    if grad_norm > eps and prev_grad_norm > eps:
                        # cos_similarity = (g_t · g_{t-1}) / (||g_t|| ||g_{t-1}||)
                        cos_similarity = torch.dot(grad.flatten(), prev_grad.flatten()) / (grad_norm * prev_grad_norm)

                        # Only apply pulse for consistent directions (cos_sim > 0)
                        consistency_weight = torch.clamp(cos_similarity, min=0.0) ** alpha

                        # Pulse strength proportional to previous gradient magnitude
                        pulse = consistency_weight * gamma * prev_grad_norm

                        # Stabilize current gradient
                        stabilizer = 1.0 + pulse / (grad_norm + eps)
                        grad = grad / stabilizer

                # Store current gradient for next iteration
                param_state['prev_grad'].copy_(p.grad)  # Store original gradient

                # Apply momentum if specified
                if momentum != 0:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(grad, alpha=1 - dampening)

                    if nesterov:
                        grad = grad.add(buf, alpha=momentum)
                    else:
                        grad = buf

                # Update parameters
                p.add_(grad, alpha=-lr)

        return loss
