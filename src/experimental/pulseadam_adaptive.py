"""
PulseAdam with adaptive gradient-normalized pulse dampening.

Applies the adaptive pulse mechanism to Adam optimizer.
"""

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List


class PulseAdamAdaptive(Optimizer):
    """
    Adam with adaptive pulse normalized by gradient magnitude.

    Standard Adam:
        m = beta1 * m + (1-beta1) * grad
        v = beta2 * v + (1-beta2) * grad²
        update = m / (sqrt(v) + eps)

    PulseAdamAdaptive:
        normalized_diff = diff / (|grad| + eps)
        pulse = gamma * normalized_diff
        denom = sqrt(v) + pulse + eps
        update = m / denom

    This adaptive formulation provides:
    - Strong dampening when gradients are small (near convergence)
    - Light dampening when gradients are large (early training)
    - Better alignment with training dynamics

    Args:
        params: iterable of parameters to optimize
        lr: learning rate (default: 1e-3)
        betas: coefficients for gradient and squared gradient (default: (0.9, 0.999))
        gamma: pulse regularization strength (default: 0.5)
        eps: term added to denominator for numerical stability (default: 1e-8)
        weight_decay: weight decay coefficient (default: 0)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), gamma=0.5,
                 eps=1e-8, weight_decay=0, *, maximize: bool = False):
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
                       weight_decay=weight_decay, maximize=maximize)
        super(PulseAdamAdaptive, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
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
            beta1, beta2 = group['betas']
            gamma = group['gamma']
            lr = group['lr']
            eps = group['eps']
            weight_decay = group['weight_decay']
            maximize = group['maximize']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad if not maximize else -p.grad

                if grad.is_sparse:
                    raise RuntimeError('PulseAdamAdaptive does not support sparse gradients')

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

                # Weight decay (decoupled, like AdamW)
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute gradient difference (pulse signal)
                diff = torch.abs(grad - previous_grad)

                # ADAPTIVE pulse: normalized by gradient magnitude
                normalized_diff = diff / (grad.abs() + 1e-8)
                pulse_term = gamma * normalized_diff

                # Adam's denominator with ADAPTIVE pulse stabilization
                denom = exp_avg_sq.sqrt() + pulse_term + eps

                # Parameter update
                p.addcdiv_(exp_avg, denom, value=-lr)

                # Store current gradient for next step
                state['previous_grad'].copy_(grad)

        return loss
