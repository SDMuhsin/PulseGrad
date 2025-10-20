from __future__ import annotations
from typing import Tuple, Callable
import torch
from torch.optim.optimizer import Optimizer


class PulseLion(Optimizer):
    """
    PulseLion: Lion optimizer with pulse-based stabilization.

    Applies the pulse stabilization concept from PulseGrad to Lion optimizer.
    The key modification: In regions of high gradient change (high diff),
    the update magnitude is scaled down by a pulse term in the denominator.

    Standard Lion update:
        update = sign(beta1 * m + (1-beta1) * g)
        p = p - lr * update

    PulseLion update:
        diff = |grad - previous_grad|
        pulse = gamma * diff²
        update = sign(beta1 * m + (1-beta1) * g) / sqrt(1 + pulse)
        p = p - lr * update

    This provides stabilization in high gradient change regions, similar to
    how PulseGrad adds gamma*diff² to the denominator.

    Args:
        params: iterable of parameters to optimize
        lr: learning rate (default: 1e-4)
        betas: coefficients for momentum (beta1, beta2) (default: (0.9, 0.99))
        gamma: pulse regularization strength (default: 0.3)
        weight_decay: weight decay coefficient (default: 0.0)

    Example:
        >>> optimizer = PulseLion(model.parameters(), lr=1e-3, gamma=0.3)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        gamma: float = 0.3,
        weight_decay: float = 0.0,
    ):
        assert lr > 0., f"Invalid learning rate: {lr}"
        assert all([0. <= beta <= 1. for beta in betas]), f"Invalid betas: {betas}"
        assert gamma >= 0., f"Invalid gamma: {gamma}"

        defaults = dict(
            lr=lr,
            betas=betas,
            gamma=gamma,
            weight_decay=weight_decay
        )

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        """
        Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            The loss if closure is provided, else None.
        """
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
                    raise RuntimeError('PulseLion does not support sparse gradients')

                lr = group['lr']
                beta1, beta2 = group['betas']
                gamma = group['gamma']
                weight_decay = group['weight_decay']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['previous_grad'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                previous_grad = state['previous_grad']

                # Weight decay (coupled, like standard Lion)
                p.data.mul_(1. - lr * weight_decay)

                # Compute gradient difference (pulse signal)
                diff = torch.abs(grad - previous_grad)

                # Interpolated momentum (Lion's approach)
                m_interp = exp_avg.clone().mul_(beta1).add(grad, alpha=1. - beta1)

                # Sign-based update (Lion's core innovation)
                update_sign = m_interp.sign()

                # Pulse stabilization: scale down updates in high-change regions
                # This is the key modification from standard Lion
                pulse_term = gamma * diff.pow(2)
                denom = (1.0 + pulse_term).sqrt()
                update = update_sign / denom

                # Parameter update
                p.add_(update, alpha=-lr)

                # Update momentum (using beta2)
                exp_avg.mul_(beta2).add_(grad, alpha=1. - beta2)

                # Store current gradient for next step
                state['previous_grad'].copy_(grad)

        return loss
