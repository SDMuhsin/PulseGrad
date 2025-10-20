"""
PulseAdaDelta with adaptive gradient-normalized pulse dampening.

Applies the adaptive pulse mechanism to AdaDelta optimizer.
"""

import torch
from torch.optim.optimizer import Optimizer


class PulseAdaDeltaAdaptive(Optimizer):
    """
    AdaDelta with adaptive pulse normalized by gradient magnitude.

    Standard AdaDelta:
        acc_delta = rho * acc_delta + (1-rho) * delta²
        acc_grad = rho * acc_grad + (1-rho) * grad²
        delta = sqrt(acc_delta + eps) / sqrt(acc_grad + eps) * grad
        update = -delta

    PulseAdaDeltaAdaptive:
        normalized_diff = diff / (|grad| + eps)
        pulse = gamma * normalized_diff
        denom = sqrt(acc_grad) + pulse + eps
        delta = sqrt(acc_delta + eps) / denom * grad
        update = -delta

    Args:
        params: iterable of parameters to optimize
        lr: coefficient that scale delta before it is applied (default: 1.0)
        rho: coefficient used for computing running averages (default: 0.9)
        gamma: pulse regularization strength (default: 0.5)
        eps: term added to denominator for numerical stability (default: 1e-6)
        weight_decay: weight decay coefficient (default: 0)
    """

    def __init__(self, params, lr=1.0, rho=0.9, gamma=0.5, eps=1e-6, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= rho < 1.0:
            raise ValueError(f"Invalid rho value: {rho}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= gamma:
            raise ValueError(f"Invalid gamma parameter: {gamma}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, rho=rho, gamma=gamma, eps=eps, weight_decay=weight_decay)
        super(PulseAdaDeltaAdaptive, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            rho = group['rho']
            gamma = group['gamma']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError('PulseAdaDeltaAdaptive does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['acc_grad'] = torch.zeros_like(p)
                    state['acc_delta'] = torch.zeros_like(p)
                    state['previous_grad'] = torch.zeros_like(p)

                acc_grad = state['acc_grad']
                acc_delta = state['acc_delta']
                previous_grad = state['previous_grad']

                state['step'] += 1

                # Weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Accumulate gradient
                acc_grad.mul_(rho).addcmul_(grad, grad, value=1 - rho)

                # Compute gradient difference (pulse signal)
                diff = torch.abs(grad - previous_grad)

                # ADAPTIVE pulse: normalized by gradient magnitude
                normalized_diff = diff / (grad.abs() + 1e-8)
                pulse_term = gamma * normalized_diff

                # AdaDelta's denominator with ADAPTIVE pulse stabilization
                denom = acc_grad.sqrt() + pulse_term + eps

                # Compute delta
                delta = (acc_delta.add(eps).sqrt() / denom) * grad

                # Accumulate delta
                acc_delta.mul_(rho).addcmul_(delta, delta, value=1 - rho)

                # Update parameters
                p.add_(delta, alpha=-lr)

                # Store current gradient for next step
                state['previous_grad'].copy_(grad)

        return loss
