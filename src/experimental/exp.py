import math
import torch
from torch.optim.optimizer import Optimizer, required

class Experimental(Optimizer):
    r"""Implements an experimental enhancement to the diffGrad algorithm.
    
    This optimizer extends diffGrad by introducing an adaptive scaling factor,
    which is updated using an additional parameter beta3. The update rule for a parameter p is:
    
        m_t = beta1 * m_{t-1} + (1 - beta1) * grad
        v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
        m̂_t = m_t / (1 - beta1^t)
        v̂_t = v_t / (1 - beta2^t)
    
        diff = |previous_grad - grad|
        k_t = beta3 * k_{t-1} + (1 - beta3) * diff
        dfc = 1 / (1 + exp(- k_t * diff))
    
        p = p - lr * (m̂_t * dfc) / (sqrt(v̂_t) + eps)
    
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate (default: 1e-3).
        betas (Tuple[float, float, float], optional): coefficients used for computing
            running averages of gradient, its square, and scaling factor (default: (0.9, 0.999, 0.95)).
        eps (float, optional): term added to the denominator for numerical stability (default: 1e-8).
        weight_decay (float, optional): L2 penalty (default: 0).
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999, 0.95), eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        beta1, beta2, beta3 = betas
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("Invalid beta1 parameter: {}".format(beta1))
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("Invalid beta2 parameter: {}".format(beta2))
        if not 0.0 <= beta3 < 1.0:
            raise ValueError("Invalid beta3 parameter: {}".format(beta3))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(Experimental, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Experimental, self).__setstate__(state)

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
            beta1, beta2, beta3 = group['betas']
            lr = group['lr']
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Experimental does not support sparse gradients')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # Previous gradient (for computing the diff)
                    state['previous_grad'] = torch.zeros_like(p.data)
                    # Scaling factor k (initialized to 1)
                    state['k'] = torch.ones_like(p.data)
                
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                previous_grad = state['previous_grad']
                k = state['k']
                
                state['step'] += 1

                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)
                
                # Update exponential moving averages
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Compute bias-corrected estimates
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Compute difference between previous and current gradient
                diff = torch.abs(previous_grad - grad)
                
                # Update the scaling factor k
                k.mul_(beta3).add_(diff, alpha=1 - beta3)
                
                # Compute the adaptive friction coefficient
                dfc = 1.0 / (1.0 + torch.exp(-k * diff))
                
                # Update previous gradient (clone to avoid in-place modification issues)
                state['previous_grad'] = grad.clone()
                
                # Modulate the first moment with dfc
                exp_avg_mod = exp_avg * dfc
                
                step_size = lr * (math.sqrt(bias_correction2) / bias_correction1)
                denom = exp_avg_sq.sqrt().add_(eps)
                
                # Parameter update
                p.data.addcdiv_(exp_avg_mod, denom, value=-step_size)
        
        return loss


