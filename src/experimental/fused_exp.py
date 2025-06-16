import math
import torch
from torch.optim.optimizer import Optimizer

# Import the compiled CUDA extension
try:
    import experimental_ext
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: CUDA extension for Experimental optimizer not found. Running in pure Python mode.")


class Experimental(Optimizer):
    r"""Implements an experimental variant of diffGrad with enhanced second moment estimation.

    This optimizer modifies the standard diffGrad update by augmenting the second moment
    with an extra term that penalizes rapid changes in the gradient, as controlled by
    the gamma parameter.
    
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): learning rate (default: 1e-3).
        betas (Tuple[float, float], optional): coefficients used for computing running averages of
            gradient and its square (default: (0.9, 0.999)).
        gamma (float, optional): regularization strength for the gradient difference term (default: 0.3).
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8).
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0).
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), gamma=0.3, eps=1e-8, weight_decay=0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        
        defaults = dict(lr=lr, betas=betas, gamma=gamma, eps=eps, weight_decay=weight_decay)
        super(Experimental, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Experimental, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        
        Arguments:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            gamma = group['gamma']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Experimental does not support sparse gradients')
                
                state = self.state[p]

                # State initialization
                if not state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['previous_grad'] = torch.zeros_like(p)
                
                state['step'] += 1
                
                # Check if we can use the fused CUDA kernel
                use_cuda_kernel = CUDA_AVAILABLE and p.is_cuda
                
                if use_cuda_kernel:
                    # --- FUSED CUDA KERNEL PATH ---
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    
                    experimental_ext.fused_step(
                        p,
                        grad,
                        state['exp_avg'],
                        state['exp_avg_sq'],
                        state['previous_grad'],
                        lr,
                        beta1,
                        beta2,
                        gamma,
                        eps,
                        weight_decay,
                        bias_correction1,
                        bias_correction2
                    )

                else:
                    # --- PURE PYTHON FALLBACK PATH ---
                    exp_avg, exp_avg_sq, previous_grad = state['exp_avg'], state['exp_avg_sq'], state['previous_grad']
                    step = state['step']
                    
                    if weight_decay != 0:
                        grad = grad.add(p, alpha=weight_decay)
                        
                    # Update biased first moment estimate.
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    
                    # Compute the absolute difference.
                    diff = torch.abs(previous_grad - grad)
                    
                    # Update biased second moment estimate.
                    new_val = grad.pow(2) + gamma * diff.pow(2)
                    exp_avg_sq.mul_(beta2).add_(new_val, alpha=1 - beta2)
                    
                    # Bias correction.
                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step
                    
                    m_hat = exp_avg / bias_correction1
                    v_hat = exp_avg_sq / bias_correction2
                    
                    # Compute diffGrad friction coefficient (dfc)
                    dfc = 1.0 / (1.0 + torch.exp(-diff))
                    
                    # Parameter update.
                    denom = v_hat.sqrt().add_(eps)
                    p.addcdiv_(m_hat * dfc, denom, value=-lr)
                    
                    # Store the current gradient for the next step.
                    state['previous_grad'].copy_(grad)
        
        return loss
