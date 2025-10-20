"""
PulseAdaBelief with adaptive gradient-normalized pulse dampening.

Applies the adaptive pulse mechanism to AdaBelief optimizer.
"""

import torch
from torch.optim.optimizer import Optimizer


class PulseAdaBeliefAdaptive(Optimizer):
    """
    AdaBelief with adaptive pulse normalized by gradient magnitude.

    Standard AdaBelief:
        m = beta1 * m + (1-beta1) * grad
        s = beta2 * s + (1-beta2) * (grad - m)²  # variance of gradients
        update = m / (sqrt(s) + eps)

    PulseAdaBeliefAdaptive:
        normalized_diff = diff / (|grad| + eps)
        pulse = gamma * normalized_diff
        denom = sqrt(s) + pulse + eps
        update = m / denom

    Args:
        params: iterable of parameters to optimize
        lr: learning rate (default: 1e-3)
        betas: coefficients for gradient and squared gradient (default: (0.9, 0.999))
        gamma: pulse regularization strength (default: 0.5)
        eps: term added to denominator for numerical stability (default: 1e-16)
        weight_decay: weight decay coefficient (default: 0)
        rectify: whether to use rectified updates (default: True)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), gamma=0.5,
                 eps=1e-16, weight_decay=0, rectify=True):
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
                       weight_decay=weight_decay, rectify=rectify)
        super(PulseAdaBeliefAdaptive, self).__init__(params, defaults)

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
            rectify = group['rectify']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError('PulseAdaBeliefAdaptive does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_var'] = torch.zeros_like(p)
                    state['previous_grad'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                exp_avg_var = state['exp_avg_var']
                previous_grad = state['previous_grad']

                state['step'] += 1

                # Decoupled weight decay
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update variance (AdaBelief's key innovation)
                grad_residual = grad - exp_avg
                exp_avg_var.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1 - beta2)

                # Compute gradient difference (pulse signal)
                diff = torch.abs(grad - previous_grad)

                # ADAPTIVE pulse: normalized by gradient magnitude
                normalized_diff = diff / (grad.abs() + 1e-8)
                pulse_term = gamma * normalized_diff

                # AdaBelief's denominator with ADAPTIVE pulse stabilization
                denom = (exp_avg_var.sqrt() + eps).add_(pulse_term)

                # Bias correction
                if rectify:
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = lr * (bias_correction2 ** 0.5) / bias_correction1
                else:
                    step_size = lr

                # Parameter update
                p.addcdiv_(exp_avg, denom, value=-step_size)

                # Store current gradient for next step
                state['previous_grad'].copy_(grad)

        return loss
