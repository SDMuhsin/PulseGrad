import math
import torch
from torch.optim.optimizer import Optimizer

class ExperimentalOptimized(Optimizer):
    r"""Implements an experimental variant of diffGrad with enhanced second moment estimation. (Optimized)

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
        super(ExperimentalOptimized, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(ExperimentalOptimized, self).__setstate__(state)

    @torch.no_grad() # Ensure no gradient tracking within the step
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad(): # Re-enable grad for closure if it's a training function
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            gamma = group['gamma']
            eps = group['eps']
            weight_decay = group['weight_decay']

            # Pre-calculate 1 - beta values
            one_minus_beta1 = 1.0 - beta1
            one_minus_beta2 = 1.0 - beta2

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad # No need for .data in modern PyTorch if not modifying inplace before autograd
                if grad.is_sparse:
                    raise RuntimeError('ExperimentalOptimized does not support sparse gradients')
                
                state = self.state[p]

                # State initialization
                if not state: # Faster to check 'len(state) == 0' or directly 'if not state:'
                    state['step'] = 0
                    # Exponential moving average of gradient values (first moment)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values (second moment)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Previous gradient for computing the gradient difference
                    state['previous_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                exp_avg, exp_avg_sq, previous_grad = state['exp_avg'], state['exp_avg_sq'], state['previous_grad']
                
                state['step'] += 1
                step = state['step']
                
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay) # p.data is implicitly handled by torch.no_grad context
                
                # Update biased first moment estimate.
                # exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg.lerp_(grad, one_minus_beta1) # Potentially more optimized for this pattern

                # Compute the absolute difference between the current and previous gradients.
                # `previous_grad` stores the `grad` from the *previous* step (potentially after weight decay)
                diff = torch.abs(previous_grad - grad)
                
                # Update biased second moment estimate with the additional regularization term.
                # new_val = grad^2 + gamma * (diff)^2
                # Using torch.square for clarity and potential minor optimization
                grad_sq = torch.square(grad)
                if gamma != 0:
                    diff_sq_reg = torch.square(diff).mul_(gamma)
                    new_val = grad_sq + diff_sq_reg
                else:
                    new_val = grad_sq
                
                # exp_avg_sq.mul_(beta2).add_(new_val, alpha=1 - beta2)
                exp_avg_sq.lerp_(new_val, one_minus_beta2)

                # Bias correction for first and second moments.
                # These are scalar ops, already efficient.
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                
                # Corrected moments
                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2
                
                # Compute the diffGrad friction coefficient (dfc)
                # Using torch.sigmoid(diff) which is equivalent to 1.0 / (1.0 + torch.exp(-diff))
                # and can be more numerically stable / faster.
                dfc = torch.sigmoid(diff) 
                
                # Parameter update.
                denom = v_hat.sqrt_().add_(eps) # In-place sqrt followed by in-place add
                                              # v_hat is only used to compute denom here, so modifying it is fine.
                
                # Numerator for the update
                update_val = m_hat * dfc
                
                # p.data.addcdiv_(m_hat * dfc, denom, value=-lr)
                p.addcdiv_(update_val, denom, value=-lr)
                
                # Store the current gradient for the next step.
                # Use copy_ to reuse memory instead of clone()
                previous_grad.copy_(grad) 
        
        return loss
