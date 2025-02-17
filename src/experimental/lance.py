import math
import torch
from torch.optim.optimizer import Optimizer

class LANCE(Optimizer):
    r"""Implements LANCE algorithm: Logistic Adaptive Nesterov Controlled Optimizer.
    
    LANCE builds on Adam by incorporating a logistic scaling factor that adaptively 
    modulates the update magnitude based on the discrepancy between the current 
    gradient and its momentum, and by leveraging a Nesterov-accelerated gradient 
    update to slightly accelerate convergence without sacrificing stability.
    
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate (default: 1e-3).
        betas (Tuple[float, float], optional): coefficients used for computing running 
            averages of gradient and its square (default: (0.9, 0.999)).
        eps (float, optional): term added to the denominator to improve numerical 
            stability (default: 1e-8).
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0).
        lam (float, optional): logistic rate parameter controlling scaling saturation 
            (default: 1.0).
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, lam=1.0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, lam=lam)
        super(LANCE, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LANCE, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        
        Arguments:
            closure (callable, optional): A closure that reevaluates the model and 
                returns the loss.
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
                    raise RuntimeError('LANCE does not support sparse gradients, please consider SparseAdam instead')
                    
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                step = state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                # Update first and second moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute bias-corrected estimates
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                m_hat = exp_avg / bias_correction1

                # Incorporate Nesterov acceleration:
                m_nesterov = beta1 * m_hat + ((1 - beta1) * grad) / bias_correction1

                # Compute denominator for adaptive update
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                # Compute logistic scaling factor based on discrepancy between grad and momentum
                scale = 1.0 / (1.0 + torch.exp(group['lam'] * torch.abs(grad - exp_avg)))

                # Compute step size with bias corrections
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # Parameter update: apply logistic scaling and Nesterov-accelerated momentum
                p.data.addcdiv_(m_nesterov, denom, value=-step_size * scale)

        return loss

