"""
PulseSGD with adaptive gradient-normalized pulse dampening.

Applies the adaptive pulse mechanism to SGD with momentum.
"""

import torch
from torch.optim.optimizer import Optimizer
from typing import Optional


class PulseSGDAdaptive(Optimizer):
    """
    SGD with momentum and adaptive pulse normalized by gradient magnitude.

    Standard SGD with momentum:
        m = momentum * m + grad
        update = m

    PulseSGDAdaptive:
        m = momentum * m + grad
        normalized_diff = diff / (|grad| + eps)
        pulse = gamma * normalized_diff
        update = m / sqrt(1 + pulse)

    This adaptive formulation provides:
    - Strong dampening when gradients are small (near convergence)
    - Light dampening when gradients are large (early training)
    - Better alignment with training dynamics

    Args:
        params: iterable of parameters to optimize
        lr: learning rate (default: 1e-3)
        momentum: momentum factor (default: 0.9)
        gamma: pulse regularization strength (default: 0.5)
        weight_decay: weight decay coefficient (default: 0)
        dampening: dampening for momentum (default: 0)
        nesterov: enables Nesterov momentum (default: False)
    """

    def __init__(self, params, lr=1e-3, momentum=0.9, gamma=0.5,
                 dampening=0, weight_decay=0, nesterov=False, *, maximize: bool = False):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= gamma:
            raise ValueError(f"Invalid gamma parameter: {gamma}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= dampening:
            raise ValueError(f"Invalid dampening value: {dampening}")

        defaults = dict(lr=lr, momentum=momentum, gamma=gamma, dampening=dampening,
                       weight_decay=weight_decay, nesterov=nesterov, maximize=maximize)

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        super(PulseSGDAdaptive, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('maximize', False)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss

        Returns:
            The loss if closure is provided, else None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            gamma = group['gamma']
            lr = group['lr']
            dampening = group['dampening']
            weight_decay = group['weight_decay']
            nesterov = group['nesterov']
            maximize = group['maximize']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad if not maximize else -p.grad

                if grad.is_sparse:
                    raise RuntimeError('PulseSGDAdaptive does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p)
                    state['previous_grad'] = torch.zeros_like(p)

                buf = state['momentum_buffer']
                previous_grad = state['previous_grad']

                # Weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Momentum update
                if momentum != 0:
                    if len(state) == 2:  # First iteration
                        buf.mul_(momentum).add_(grad)
                    else:
                        buf.mul_(momentum).add_(grad, alpha=1 - dampening)

                    if nesterov:
                        grad = grad.add(buf, alpha=momentum)
                    else:
                        grad = buf
                else:
                    grad = grad

                # Compute gradient difference (pulse signal)
                diff = torch.abs(p.grad - previous_grad)

                # ADAPTIVE pulse: normalized by gradient magnitude
                normalized_diff = diff / (p.grad.abs() + 1e-8)
                pulse_term = gamma * normalized_diff

                # SGD update with ADAPTIVE pulse stabilization
                denom = (1.0 + pulse_term).sqrt()
                update = grad / denom

                # Parameter update
                p.add_(update, alpha=-lr)

                # Store current gradient for next step
                state['previous_grad'].copy_(p.grad)

        return loss
