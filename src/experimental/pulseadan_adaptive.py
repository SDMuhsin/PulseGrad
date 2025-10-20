"""
PulseAdan with adaptive gradient-normalized pulse dampening.

Applies the adaptive pulse mechanism to Adan (Adaptive Nesterov Momentum) optimizer.
"""

import torch
from torch.optim.optimizer import Optimizer


class PulseAdanAdaptive(Optimizer):
    """
    Adan with adaptive pulse normalized by gradient magnitude.

    Standard Adan:
        Uses three moments: gradient, gradient difference, and second moment
        m_t = beta1 * m_{t-1} + (1-beta1) * grad
        v_t = beta2 * v_{t-1} + (1-beta2) * (grad - grad_{t-1})
        n_t = beta3 * n_{t-1} + (1-beta3) * grad²
        update = (m + beta2 * v) / sqrt(n + eps)

    PulseAdanAdaptive:
        normalized_diff = diff / (|grad| + eps)
        pulse = gamma * normalized_diff
        denom = sqrt(n) + pulse + eps
        update = (m + beta2 * v) / denom

    Args:
        params: iterable of parameters to optimize
        lr: learning rate (default: 1e-3)
        betas: coefficients (default: (0.98, 0.92, 0.99))
        gamma: pulse regularization strength (default: 0.5)
        eps: term added to denominator for numerical stability (default: 1e-8)
        weight_decay: weight decay coefficient (default: 0)
    """

    def __init__(self, params, lr=1e-3, betas=(0.98, 0.92, 0.99), gamma=0.5,
                 eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 2: {betas[2]}")
        if not 0.0 <= gamma:
            raise ValueError(f"Invalid gamma parameter: {gamma}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, gamma=gamma, eps=eps, weight_decay=weight_decay)
        super(PulseAdanAdaptive, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2, beta3 = group['betas']
            gamma = group['gamma']
            lr = group['lr']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError('PulseAdanAdaptive does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)  # First moment (m)
                    state['exp_avg_diff'] = torch.zeros_like(p)  # Gradient difference moment (v)
                    state['exp_avg_sq'] = torch.zeros_like(p)  # Second moment (n)
                    state['previous_grad'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                exp_avg_diff = state['exp_avg_diff']
                exp_avg_sq = state['exp_avg_sq']
                previous_grad = state['previous_grad']

                state['step'] += 1

                # Weight decay (decoupled)
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                # Compute gradient difference
                grad_diff = grad - previous_grad

                # Update first moment (EMA of gradient)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update gradient difference moment (EMA of gradient difference)
                exp_avg_diff.mul_(beta2).add_(grad_diff, alpha=1 - beta2)

                # Update second moment (EMA of squared gradient)
                exp_avg_sq.mul_(beta3).addcmul_(grad, grad, value=1 - beta3)

                # Compute pulse difference (for adaptive pulse)
                diff = torch.abs(grad_diff)

                # ADAPTIVE pulse: normalized by gradient magnitude
                normalized_diff = diff / (grad.abs() + 1e-8)
                pulse_term = gamma * normalized_diff

                # Adan's denominator with ADAPTIVE pulse stabilization
                denom = exp_avg_sq.sqrt() + pulse_term + eps

                # Nesterov-style update combining m and v
                update_term = exp_avg.add(exp_avg_diff, alpha=beta2)

                # Parameter update
                p.addcdiv_(update_term, denom, value=-lr)

                # Store current gradient for next step
                state['previous_grad'].copy_(grad)

        return loss
