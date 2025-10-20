"""
PulseMADGrad with adaptive gradient-normalized pulse dampening.

Applies the adaptive pulse mechanism to MADGrad (Momentum-Adaptive Dual Averaged Gradient) optimizer.
"""

import torch
from torch.optim.optimizer import Optimizer


class PulseMADGradAdaptive(Optimizer):
    """
    MADGrad with adaptive pulse normalized by gradient magnitude.

    Standard MADGrad:
        Uses momentum and dual averaging with adaptive step sizes
        s_t = s_{t-1} + grad
        v_t = v_{t-1} + grad²
        z_t = z_{t-1} + grad / sqrt(v_t + eps)
        update = z_t

    PulseMADGradAdaptive:
        normalized_diff = diff / (|grad| + eps)
        pulse = gamma * normalized_diff
        Modified denominator with pulse stabilization

    Args:
        params: iterable of parameters to optimize
        lr: learning rate (default: 1e-2)
        momentum: momentum factor (default: 0.9)
        gamma: pulse regularization strength (default: 0.5)
        eps: term added to denominator for numerical stability (default: 1e-6)
        weight_decay: weight decay coefficient (default: 0)
    """

    def __init__(self, params, lr=1e-2, momentum=0.9, gamma=0.5,
                 eps=1e-6, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= gamma:
            raise ValueError(f"Invalid gamma parameter: {gamma}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, momentum=momentum, gamma=gamma, eps=eps, weight_decay=weight_decay)
        super(PulseMADGradAdaptive, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            gamma = group['gamma']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError('PulseMADGradAdaptive does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['grad_sum_sq'] = torch.zeros_like(p)
                    state['s'] = torch.zeros_like(p)  # Dual averaging state
                    state['previous_grad'] = torch.zeros_like(p)

                grad_sum_sq = state['grad_sum_sq']
                s = state['s']
                previous_grad = state['previous_grad']

                state['step'] += 1

                # Weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Update sum of squares
                grad_sum_sq.addcmul_(grad, grad, value=1)

                # Compute gradient difference (pulse signal)
                diff = torch.abs(grad - previous_grad)

                # ADAPTIVE pulse: normalized by gradient magnitude
                normalized_diff = diff / (grad.abs() + 1e-8)
                pulse_term = gamma * normalized_diff

                # MADGrad's denominator with ADAPTIVE pulse stabilization
                rms = grad_sum_sq.sqrt()
                denom = rms + pulse_term + eps

                # Update dual averaging state with momentum
                z = grad / denom
                s.mul_(momentum).add_(z, alpha=1)

                # Parameter update
                p.add_(s, alpha=-lr)

                # Store current gradient for next step
                state['previous_grad'].copy_(grad)

        return loss
