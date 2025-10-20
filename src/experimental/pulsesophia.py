import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Optional


class PulseSophia(Optimizer):
    """
    PulseSophia: Sophia-G optimizer with pulse-based stabilization.

    Combines Sophia's Hessian-based curvature adaptation with PulseGrad's
    gradient change dampening for enhanced stability.

    Standard Sophia update:
        m = beta1 * m + (1-beta1) * grad
        h = beta2 * h + (1-beta2) * grad²
        ratio = |m| / (rho * bs * h + eps)
        p -= lr * sign(m) * ratio

    PulseSophia update:
        diff = |grad - previous_grad|
        pulse = gamma * diff²
        ratio = |m| / (rho * bs * h + pulse + eps)
        p -= lr * sign(m) * ratio

    The pulse term provides additional dampening in regions of high
    gradient change, complementing Sophia's curvature-based adaptation.

    Args:
        params: iterable of parameters to optimize
        lr: learning rate (default: 1e-4)
        betas: coefficients for momentum and hessian (default: (0.965, 0.99))
        rho: hessian scaling factor (default: 0.04)
        gamma: pulse regularization strength (default: 0.3)
        weight_decay: weight decay coefficient (default: 1e-1)
    """

    def __init__(self, params, lr=1e-4, betas=(0.965, 0.99), rho=0.04,
                 gamma=0.3, weight_decay=1e-1, *, maximize: bool = False):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= rho:
            raise ValueError(f"Invalid rho parameter: {rho}")
        if not 0.0 <= gamma:
            raise ValueError(f"Invalid gamma parameter: {gamma}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, rho=rho, gamma=gamma,
                        weight_decay=weight_decay, maximize=maximize)
        super(PulseSophia, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('maximize', False)

    @torch.no_grad()
    def update_hessian(self):
        """Update the Hessian diagonal estimate."""
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = torch.tensor(0.)
                    state['exp_avg'] = torch.zeros_like(p)
                    state['hessian'] = torch.zeros_like(p)
                    state['previous_grad'] = torch.zeros_like(p)

                if 'hessian' not in state:
                    state['hessian'] = torch.zeros_like(p)

                # Update Hessian diagonal: h = beta2 * h + (1-beta2) * grad²
                state['hessian'].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

    @torch.no_grad()
    def update_exp_avg(self):
        """Update the exponential moving average of gradients (momentum)."""
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                # Update momentum: m = beta1 * m + (1-beta1) * grad
                state['exp_avg'].mul_(beta1).add_(p.grad, alpha=1 - beta1)

    @torch.no_grad()
    def step(self, closure=None, bs=5120):
        """
        Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss
            bs: Batch size for Hessian scaling (default: 5120, Sophia's default)

        Returns:
            The loss if closure is provided, else None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Update Hessian and momentum
        self.update_hessian()
        self.update_exp_avg()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            rho = group['rho']
            gamma = group['gamma']
            lr = group['lr']
            weight_decay = group['weight_decay']
            maximize = group['maximize']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad if not maximize else -p.grad

                if grad.is_sparse:
                    raise RuntimeError('PulseSophia does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = torch.tensor(0.)
                    state['exp_avg'] = torch.zeros_like(p)
                    state['hessian'] = torch.zeros_like(p)
                    state['previous_grad'] = torch.zeros_like(p)

                if 'hessian' not in state:
                    state['hessian'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                hess = state['hessian']
                previous_grad = state['previous_grad']

                # Update step
                state['step'] += 1

                # Weight decay (coupled, like Sophia)
                p.mul_(1 - lr * weight_decay)

                # Compute gradient difference (pulse signal)
                diff = torch.abs(grad - previous_grad)

                # Pulse stabilization term
                pulse_term = gamma * diff.pow(2)

                # Sophia's adaptive ratio with pulse stabilization
                # Original: ratio = |m| / (rho * bs * h + eps)
                # PulseSophia: ratio = |m| / (rho * bs * h + pulse + eps)
                denominator = rho * bs * hess + pulse_term + 1e-15
                ratio = (exp_avg.abs() / denominator).clamp(None, 1)

                # Parameter update: p -= lr * sign(m) * ratio
                p.addcmul_(exp_avg.sign(), ratio, value=-lr)

                # Store current gradient for next step
                state['previous_grad'].copy_(grad)

        return loss
