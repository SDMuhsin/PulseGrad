import torch
from torch.optim.optimizer import Optimizer
import math

class CATS(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99, 0.95), 
                 gamma=1.0, lambda_=0.1, eps=1e-8):
        defaults = dict(lr=lr, betas=betas, gamma=gamma,
                        lambda_=lambda_, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self,closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum'] = torch.zeros_like(p)
                    state['mu'] = torch.zeros_like(p)
                    state['variance'] = torch.zeros_like(p)
                
                beta1, beta2, beta3 = group['betas']
                state['step'] += 1
                t = state['step']
                
                # Update momentum
                state['momentum'].mul_(beta1).add_(grad, alpha=1-beta1)
                
                # Update mean estimate
                state['mu'].mul_(beta3).add_(grad, alpha=1-beta3)
                
                # Update variance estimate
                delta = grad - state['mu']
                state['variance'].mul_(beta2).addcmul_(delta, delta, value=1-beta2)
                
                # Adaptive beta1
                beta1_t = 1 - 1/(t**0.9 + 10)
                
                # Compute step size
                v_sqrt = state['variance'].sqrt().add_(group['eps'])
                eta = group['lr'] / v_sqrt
                eta *= torch.tanh(group['lambda_'] * v_sqrt)
                
                # Direction selection
                direction = torch.sign(state['momentum'])
                
                # Apply update with trust region
                step = eta * direction
                max_step = group['gamma'] * v_sqrt
                step.clamp_(-max_step, max_step)
                
                p.add_(-step)
                
                # Post-update momentum correction
                state['momentum'].mul_(beta1_t / beta1)

