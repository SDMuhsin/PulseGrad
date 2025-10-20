"""
Tuned variants of PulseLion with stronger pulse dampening.

Based on diagnostics showing original pulse (gamma=0.3, squared) is too weak.
"""

from __future__ import annotations
from typing import Tuple, Callable
import torch
from torch.optim.optimizer import Optimizer


class PulseLionStrong(Optimizer):
    """
    PulseLion with much stronger gamma (default: 10.0 instead of 0.3)

    Diagnostic finding: Original pulse term ~0.01, denominator ~1.005
    With gamma=10: pulse term ~0.3, denominator ~1.14 (significant dampening)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        gamma: float = 10.0,  # 33x stronger than original!
        weight_decay: float = 0.0,
    ):
        assert lr > 0., f"Invalid learning rate: {lr}"
        assert all([0. <= beta <= 1. for beta in betas]), f"Invalid betas: {betas}"
        assert gamma >= 0., f"Invalid gamma: {gamma}"

        defaults = dict(lr=lr, betas=betas, gamma=gamma, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('PulseLionStrong does not support sparse gradients')

                lr = group['lr']
                beta1, beta2 = group['betas']
                gamma = group['gamma']
                weight_decay = group['weight_decay']

                state = self.state[p]

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['previous_grad'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                previous_grad = state['previous_grad']

                # Weight decay
                p.data.mul_(1. - lr * weight_decay)

                # Compute gradient difference
                diff = torch.abs(grad - previous_grad)

                # Interpolated momentum
                m_interp = exp_avg.clone().mul_(beta1).add(grad, alpha=1. - beta1)

                # Sign-based update
                update_sign = m_interp.sign()

                # STRONG pulse stabilization (gamma=10 default)
                pulse_term = gamma * diff.pow(2)
                denom = (1.0 + pulse_term).sqrt()
                update = update_sign / denom

                # Parameter update
                p.add_(update, alpha=-lr)

                # Update momentum
                exp_avg.mul_(beta2).add_(grad, alpha=1. - beta2)

                # Store current gradient
                state['previous_grad'].copy_(grad)

        return loss


class PulseLionLinear(Optimizer):
    """
    PulseLion with LINEAR pulse term instead of squared.

    Formula: update = sign(m) / sqrt(1 + gamma * |diff|)
    Instead of: update = sign(m) / sqrt(1 + gamma * diff²)

    Linear scaling is less aggressive but more predictable.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        gamma: float = 1.0,
        weight_decay: float = 0.0,
    ):
        assert lr > 0., f"Invalid learning rate: {lr}"
        assert all([0. <= beta <= 1. for beta in betas]), f"Invalid betas: {betas}"
        assert gamma >= 0., f"Invalid gamma: {gamma}"

        defaults = dict(lr=lr, betas=betas, gamma=gamma, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('PulseLionLinear does not support sparse gradients')

                lr = group['lr']
                beta1, beta2 = group['betas']
                gamma = group['gamma']
                weight_decay = group['weight_decay']

                state = self.state[p]

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['previous_grad'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                previous_grad = state['previous_grad']

                # Weight decay
                p.data.mul_(1. - lr * weight_decay)

                # Compute gradient difference
                diff = torch.abs(grad - previous_grad)

                # Interpolated momentum
                m_interp = exp_avg.clone().mul_(beta1).add(grad, alpha=1. - beta1)

                # Sign-based update
                update_sign = m_interp.sign()

                # LINEAR pulse stabilization
                pulse_term = gamma * diff  # Linear, not squared!
                denom = (1.0 + pulse_term).sqrt()
                update = update_sign / denom

                # Parameter update
                p.add_(update, alpha=-lr)

                # Update momentum
                exp_avg.mul_(beta2).add_(grad, alpha=1. - beta2)

                # Store current gradient
                state['previous_grad'].copy_(grad)

        return loss


class PulseLionAdaptive(Optimizer):
    """
    PulseLion with adaptive pulse normalized by gradient magnitude.

    Formula: pulse = gamma * diff / (|grad| + eps)

    This makes pulse strength relative to gradient size,
    providing stronger dampening when gradients are small.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        gamma: float = 0.5,
        weight_decay: float = 0.0,
    ):
        assert lr > 0., f"Invalid learning rate: {lr}"
        assert all([0. <= beta <= 1. for beta in betas]), f"Invalid betas: {betas}"
        assert gamma >= 0., f"Invalid gamma: {gamma}"

        defaults = dict(lr=lr, betas=betas, gamma=gamma, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('PulseLionAdaptive does not support sparse gradients')

                lr = group['lr']
                beta1, beta2 = group['betas']
                gamma = group['gamma']
                weight_decay = group['weight_decay']

                state = self.state[p]

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['previous_grad'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                previous_grad = state['previous_grad']

                # Weight decay
                p.data.mul_(1. - lr * weight_decay)

                # Compute gradient difference
                diff = torch.abs(grad - previous_grad)

                # Interpolated momentum
                m_interp = exp_avg.clone().mul_(beta1).add(grad, alpha=1. - beta1)

                # Sign-based update
                update_sign = m_interp.sign()

                # ADAPTIVE pulse: normalized by gradient magnitude
                normalized_diff = diff / (grad.abs() + 1e-8)
                pulse_term = gamma * normalized_diff
                denom = (1.0 + pulse_term).sqrt()
                update = update_sign / denom

                # Parameter update
                p.add_(update, alpha=-lr)

                # Update momentum
                exp_avg.mul_(beta2).add_(grad, alpha=1. - beta2)

                # Store current gradient
                state['previous_grad'].copy_(grad)

        return loss
