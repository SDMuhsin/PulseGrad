import torch
from torch.optim.optimizer import Optimizer
import math

class HASOpt(Optimizer):
    """
    Implements the Hamiltonian Adaptive Symplectic Optimizer (HASOpt).
    """
    def __init__(self, params, lr=1e-3, betas=(0.95, 0.99), gamma=0.1, 
                 eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, gamma=gamma,
                        eps=eps, weight_decay=weight_decay)
        super(HASOpt, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('HASOpt does not support sparse gradients')

                state = self.state[p]
                
                # Apply weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                beta1, beta2 = group['betas']
                lr = group['lr']
                eps = group['eps']
                gamma = group['gamma']

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    state['v'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    state['mass'] = torch.ones_like(p.data, memory_format=torch.preserve_format)
                    state['grad_buffer'] = torch.zeros(5, *p.data.shape, device=p.device)
                    state['momentum'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                m, v, mass, grad_buf, momentum = (
                    state['m'], state['v'], state['mass'], state['grad_buffer'], state['momentum']
                )
                
                state['step'] += 1
                step = state['step']

                # Update gradient buffer
                grad_buf = torch.roll(grad_buf, shifts=1, dims=0)
                grad_buf[0].copy_(grad)

                # Covariance calculation
                if step > 5:
                    grad_flat = grad_buf.view(5, -1)
                    cov = torch.cov(grad_flat)
                    cov_penalty = torch.norm(cov, p='nuc')
                else:
                    # FIX: Ensure cov_penalty is a tensor on the correct device.
                    cov_penalty = torch.tensor(0.0, device=grad.device)

                # Update biased first and second moment estimates
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Mass matrix adaptation
                mass_update = 1.0 / (v.sqrt() + eps) * torch.exp(-gamma * cov_penalty)
                mass.mul_(0.9).add_(mass_update, alpha=0.1)

                # Symplectic update
                # 1. Momentum half-step update
                scaled_momentum = momentum * torch.exp(-0.5 * lr * mass * grad.sign())
                
                # 2. Parameter update with rescaled momentum
                direction = scaled_momentum.sign()
                magnitude = scaled_momentum.abs().pow(0.5)
                p.data.addcmul_(direction, magnitude, value=-lr)
                
                # 3. Momentum correction (second half-step)
                momentum.data = scaled_momentum.mul(
                    torch.exp(-0.5 * lr * grad)
                ).addcmul_(grad, mass, value=-lr)

        return loss

