import math
import torch
from torch.optim.optimizer import Optimizer

class COSMIC(Optimizer):
    r"""Implements COSMIC: COsine Similarity Momentum-based Adaptive Gradient Optimizer.
    It modifies Adam by modulating the first moment using the cosine similarity between
    the previous momentum and the current gradient. This helps amplify updates when the 
    gradient direction is consistent and dampen them when the direction fluctuates.
    
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate (default: 1e-3).
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999)).
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8).
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0).
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(COSMIC, self).__init__(params, defaults)
        
        # Additional parameters for COSMIC
        self.epsilon_prime = 0.5  # Lower bound for modulation factor psi.
        self.epsilon0 = 1e-8      # Small constant to avoid division by zero in cosine similarity.

    def __setstate__(self, state):
        super(COSMIC, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('COSMIC does not support sparse gradients, please consider SparseAdam instead')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values (first moment)
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values (second moment)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # Previous momentum for cosine similarity modulation
                    state['previous_m'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq, previous_m = state['exp_avg'], state['exp_avg_sq'], state['previous_m']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)
                
                # Update biased first and second moment estimates
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Compute the cosine similarity modulation factor psi elementwise.
                # For each element:
                #   If |previous_m| or |grad| is too small, set psi = 1.
                #   Otherwise, compute cos_sim = (previous_m * grad) / (|previous_m|*|grad| + epsilon0),
                #   clip it to non-negative values, and then
                #   psi = epsilon_prime + (1 - epsilon_prime) * cos_sim.
                cos_sim = torch.zeros_like(grad)
                mask = (previous_m.abs() > self.epsilon0) & (grad.abs() > self.epsilon0)
                cos_sim[mask] = (previous_m[mask] * grad[mask]) / (previous_m[mask].abs() * grad[mask].abs() + self.epsilon0)
                cos_sim = torch.clamp(cos_sim, min=0.0)
                psi = self.epsilon_prime + (1.0 - self.epsilon_prime) * cos_sim
                
                # Modulate the first moment with psi.
                modulated_exp_avg = exp_avg * psi
                
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                
                p.data.addcdiv_(-step_size, modulated_exp_avg, denom)
                
                # Update previous momentum for the next iteration.
                state['previous_m'] = exp_avg.clone()
        
        return loss

