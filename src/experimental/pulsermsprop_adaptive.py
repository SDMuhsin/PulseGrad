"""
PulseRMSprop with adaptive gradient-normalized pulse dampening.

Applies the adaptive pulse mechanism to RMSprop optimizer.
"""

import torch
from torch.optim.optimizer import Optimizer


class PulseRMSpropAdaptive(Optimizer):
    """
    RMSprop with adaptive pulse normalized by gradient magnitude.

    Standard RMSprop:
        v = alpha * v + (1-alpha) * grad²
        if momentum > 0:
            buf = momentum * buf + grad / sqrt(v + eps)
            update = buf
        else:
            update = grad / sqrt(v + eps)

    PulseRMSpropAdaptive:
        normalized_diff = diff / (|grad| + eps)
        pulse = gamma * normalized_diff
        denom = sqrt(v) + pulse + eps
        update = grad / denom

    Args:
        params: iterable of parameters to optimize
        lr: learning rate (default: 1e-2)
        alpha: smoothing constant (default: 0.99)
        momentum: momentum factor (default: 0)
        gamma: pulse regularization strength (default: 0.5)
        eps: term added to denominator for numerical stability (default: 1e-8)
        weight_decay: weight decay coefficient (default: 0)
        centered: if True, compute centered RMSprop (default: False)
    """

    def __init__(self, params, lr=1e-2, alpha=0.99, momentum=0, gamma=0.5,
                 eps=1e-8, weight_decay=0, centered=False):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= alpha < 1.0:
            raise ValueError(f"Invalid alpha value: {alpha}")
        if not 0.0 <= momentum:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= gamma:
            raise ValueError(f"Invalid gamma parameter: {gamma}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, alpha=alpha, momentum=momentum, gamma=gamma,
                       eps=eps, weight_decay=weight_decay, centered=centered)
        super(PulseRMSpropAdaptive, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            alpha = group['alpha']
            momentum = group['momentum']
            gamma = group['gamma']
            eps = group['eps']
            weight_decay = group['weight_decay']
            centered = group['centered']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError('PulseRMSpropAdaptive does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p)
                    state['previous_grad'] = torch.zeros_like(p)
                    if momentum > 0:
                        state['momentum_buffer'] = torch.zeros_like(p)
                    if centered:
                        state['grad_avg'] = torch.zeros_like(p)

                square_avg = state['square_avg']
                previous_grad = state['previous_grad']

                state['step'] += 1

                # Weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Update square average
                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

                # Centered variant
                if centered:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)
                    avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).sqrt()
                else:
                    avg = square_avg.sqrt()

                # Compute gradient difference (pulse signal)
                diff = torch.abs(grad - previous_grad)

                # ADAPTIVE pulse: normalized by gradient magnitude
                normalized_diff = diff / (grad.abs() + 1e-8)
                pulse_term = gamma * normalized_diff

                # RMSprop's denominator with ADAPTIVE pulse stabilization
                denom = avg + pulse_term + eps

                if momentum > 0:
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).addcdiv_(grad, denom)
                    p.add_(buf, alpha=-lr)
                else:
                    p.addcdiv_(grad, denom, value=-lr)

                # Store current gradient for next step
                state['previous_grad'].copy_(grad)

        return loss
