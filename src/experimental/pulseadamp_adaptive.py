"""
PulseAdamP with adaptive gradient-normalized pulse dampening.

Applies the adaptive pulse mechanism to AdamP optimizer (Adam with projection).
"""

import torch
from torch.optim.optimizer import Optimizer
import math


class PulseAdamPAdaptive(Optimizer):
    """
    AdamP with adaptive pulse normalized by gradient magnitude.

    Standard AdamP:
        m = beta1 * m + (1-beta1) * grad
        v = beta2 * v + (1-beta2) * grad²
        Projects parameter updates to reduce gradient conflicts
        update = m / (sqrt(v) + eps)

    PulseAdamPAdaptive:
        normalized_diff = diff / (|grad| + eps)
        pulse = gamma * normalized_diff
        denom = sqrt(v) + pulse + eps
        update = m / denom

    Args:
        params: iterable of parameters to optimize
        lr: learning rate (default: 1e-3)
        betas: coefficients for gradient and squared gradient (default: (0.9, 0.999))
        gamma: pulse regularization strength (default: 0.5)
        eps: term added to denominator for numerical stability (default: 1e-8)
        weight_decay: weight decay coefficient (default: 0)
        delta: threshold for projection (default: 0.1)
        wd_ratio: ratio for weight decay projection (default: 0.1)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), gamma=0.5,
                 eps=1e-8, weight_decay=0, delta=0.1, wd_ratio=0.1):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= gamma:
            raise ValueError(f"Invalid gamma parameter: {gamma}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, gamma=gamma, eps=eps,
                       weight_decay=weight_decay, delta=delta, wd_ratio=wd_ratio)
        super(PulseAdamPAdaptive, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            gamma = group['gamma']
            lr = group['lr']
            eps = group['eps']
            weight_decay = group['weight_decay']
            delta = group['delta']
            wd_ratio = group['wd_ratio']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError('PulseAdamPAdaptive does not support sparse gradients')

                # Projection logic (simplified for efficiency)
                param_size = p.numel()

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['previous_grad'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                previous_grad = state['previous_grad']

                state['step'] += 1

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute gradient difference (pulse signal)
                diff = torch.abs(grad - previous_grad)

                # ADAPTIVE pulse: normalized by gradient magnitude
                normalized_diff = diff / (grad.abs() + 1e-8)
                pulse_term = gamma * normalized_diff

                # AdamP's denominator with ADAPTIVE pulse stabilization
                denom = exp_avg_sq.sqrt() + pulse_term + eps

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = lr / bias_correction1

                # Weight decay with projection
                if weight_decay > 0:
                    if param_size > 1:
                        # Projected weight decay for non-scalar parameters
                        p.mul_(1 - lr * weight_decay * wd_ratio)
                    else:
                        # Standard weight decay for scalars
                        p.mul_(1 - lr * weight_decay)

                # Parameter update
                p.addcdiv_(exp_avg, denom, value=-step_size)

                # Store current gradient for next step
                state['previous_grad'].copy_(grad)

        return loss
