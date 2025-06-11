import torch
from torch.optim.optimizer import Optimizer


class ExperimentalV3(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99, 0.999), eps=1e-8, gamma=0.1, weight_decay=0):
        """
        DualMomentum Adaptive Optimization (DAMO)
        
        Arguments:
            params (iterable): iterable of parameters to optimize or dicts defining parameter groups
            lr (float, optional): learning rate (default: 1e-3)
            betas (tuple, optional): coefficients used for computing running averages of gradient sign, 
                                    magnitude, and square (default: (0.9, 0.99, 0.999))
            eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
            gamma (float, optional): factor for the radial component correction (default: 0.1)
            weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 2: {betas[2]}")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"Invalid gamma parameter: {gamma}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, gamma=gamma, weight_decay=weight_decay)
        super(ExperimentalV3, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(ExperimentalV3, self).__setstate__(state)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
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
                    raise RuntimeError('DAMO does not support sparse gradients')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Sign momentum
                    state['sign_momentum'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Magnitude momentum
                    state['mag_momentum'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Second moment
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Max of second moment
                    state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                sign_momentum = state['sign_momentum']
                mag_momentum = state['mag_momentum']
                exp_avg_sq = state['exp_avg_sq']
                max_exp_avg_sq = state['max_exp_avg_sq']
                
                beta1, beta2, beta3 = group['betas']
                step = state['step'] = state['step'] + 1
                
                # Weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                
                # Update sign momentum: beta1 * s_{t-1} + (1 - beta1) * sign(g_t)
                sign_momentum.mul_(beta1).add_(torch.sign(grad), alpha=1 - beta1)
                
                # Update magnitude momentum: beta2 * m_{t-1} + (1 - beta2) * |g_t|
                mag_momentum.mul_(beta2).add_(grad.abs(), alpha=1 - beta2)
                
                # Update second moment: beta3 * v_{t-1} + (1 - beta3) * g_t^2
                exp_avg_sq.mul_(beta3).addcmul_(grad, grad, value=1 - beta3)
                
                # Update max of second moment: max(v_max_{t-1}, v_t)
                torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                bias_correction3 = 1 - beta3 ** step
                
                # Compute corrected sign momentum: s_t / (1 - beta1^t)
                sign_momentum_corrected = sign_momentum.clone().div_(bias_correction1)
                
                # Compute corrected magnitude momentum: m_t / (1 - beta2^t)
                mag_momentum_corrected = mag_momentum.clone().div_(bias_correction2)
                
                # Compute corrected max second moment: v_max_t / (1 - beta3^t)
                max_exp_avg_sq_corrected = max_exp_avg_sq.clone().div_(bias_correction3)
                
                # Compute update: s_t_corrected * m_t_corrected / sqrt(v_max_t_corrected + eps)
                denom = max_exp_avg_sq_corrected.sqrt().add_(group['eps'])
                update = sign_momentum_corrected * mag_momentum_corrected / denom
                
                # Apply radial correction if gamma > 0
                if group['gamma'] > 0:
                    p_norm_sq = torch.sum(p * p).add_(group['eps'])
                    update_dot_p = torch.sum(update * p)
                    radial = update_dot_p / p_norm_sq * p
                    update = update - group['gamma'] * radial
                
                # Update parameters: theta_{t+1} = theta_t - lr * update
                p.add_(update, alpha=-group['lr'])
        
        return loss

