import math
import torch
from torch.optim.optimizer import Optimizer

class DCOSMIC(Optimizer):
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

class DCOSMICV2(Optimizer):
    r"""Implements COSMIC: COsine Similarity Momentum-based Adaptive Gradient Optimizer.
    
    COSMIC modifies Adam by modulating the first moment using the cosine similarity between
    the previous momentum and the current gradient. This modulation amplifies updates when the 
    gradient direction is consistent and dampens them when the direction fluctuates.
    
    In addition to the core functionality, two stabilization controls are provided:
      1. psi_smoothing: An exponential moving average (EMA) is applied to the raw cosine-based
         modulation factor, reducing its volatility.
      2. max_update_norm: Optionally clips the norm of the parameter update to a specified maximum.
    
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate (default: 1e-3).
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999)).
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8).
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0).
        psi_smoothing (float, optional): smoothing coefficient for psi (EMA factor, default: 0.9).
        max_update_norm (float, optional): maximum norm allowed for the update (default: None, no clipping).
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
                 psi_smoothing=0.9, max_update_norm=1.0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        psi_smoothing=psi_smoothing, max_update_norm=max_update_norm)
        super(COSMIC, self).__init__(params, defaults)
        
        # Additional constants for the cosine similarity modulation
        self.epsilon_prime = 0.5  # Lower bound for the modulation factor psi.
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
            psi_smoothing = group.get('psi_smoothing', 0.9)
            max_update_norm = group.get('max_update_norm', None)
            
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
                    # Previous momentum for cosine similarity computation
                    state['previous_m'] = torch.zeros_like(p.data)
                    # EMA of the raw psi modulation factor for stabilization
                    state['psi_avg'] = torch.ones_like(p.data)
                
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                previous_m = state['previous_m']
                psi_avg = state['psi_avg']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)
                
                # Update biased first and second moment estimates (Adam style)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Compute the raw cosine similarity modulation factor psi elementwise.
                # For elements where either previous_m or grad is nearly zero, we default psi to 1.
                cos_sim = torch.zeros_like(grad)
                mask = (previous_m.abs() > self.epsilon0) & (grad.abs() > self.epsilon0)
                cos_sim[mask] = (previous_m[mask] * grad[mask]) / (previous_m[mask].abs() * grad[mask].abs() + self.epsilon0)
                # Ensure non-negativity (core functionality: modulate more when directions align)
                cos_sim = torch.clamp(cos_sim, min=0.0)
                raw_psi = self.epsilon_prime + (1.0 - self.epsilon_prime) * cos_sim
                
                # Stabilization Control 1: Smooth the psi values with an exponential moving average.
                psi_avg.mul_(psi_smoothing).add_(1 - psi_smoothing, raw_psi)
                
                # Use the smoothed psi in the momentum modulation.
                modulated_exp_avg = exp_avg * psi_avg
                
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                
                # Compute the update.
                update = -step_size * (modulated_exp_avg / denom)
                
                # Stabilization Control 2: Optionally clip the update norm.
                if max_update_norm is not None:
                    update_norm = update.norm()
                    if update_norm > max_update_norm:
                        update = update * (max_update_norm / (update_norm + 1e-6))
                
                p.data.add_(update)
                
                # Update the previous momentum for the next iteration.
                state['previous_m'] = exp_avg.clone()
        
        return loss

class DCOSMICV3(Optimizer):
    r"""Implements COSMIC: COsine Similarity Momentum-based Adaptive Gradient Optimizer.
    
    COSMIC modifies Adam by modulating the first moment using the cosine similarity
    between the previous momentum and the current gradient. This modulation amplifies
    updates when the gradient direction is consistent and dampens them when the
    direction fluctuates. In addition, an optional update clipping mechanism is provided
    to guard against excessively large steps when using higher learning rates.
    
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate (default: 1e-3).
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999)).
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8).
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0).
        max_update_norm (float, optional): maximum allowed norm of the parameter update.
            When provided, the computed update is clipped to this value (default: None).
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, max_update_norm=None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, max_update_norm=max_update_norm)
        super(COSMIC, self).__init__(params, defaults)
        
        # Additional parameters for COSMIC's cosine similarity modulation.
        self.epsilon_prime = 0.5  # Lower bound for modulation factor psi.
        self.epsilon0 = 1e-8      # Small constant to avoid division by zero in cosine similarity.

    def __setstate__(self, state):
        super(COSMIC, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        
        Arguments:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
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
                
                # State initialization.
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values (first moment).
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values (second moment).
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # Previous momentum for cosine similarity computation.
                    state['previous_m'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq, previous_m = state['exp_avg'], state['exp_avg_sq'], state['previous_m']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)
                
                # Update biased first and second moment estimates.
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Compute the cosine similarity modulation factor psi elementwise.
                # If either |previous_m| or |grad| is too small, default psi to 1.
                cos_sim = torch.zeros_like(grad)
                mask = (previous_m.abs() > self.epsilon0) & (grad.abs() > self.epsilon0)
                cos_sim[mask] = (previous_m[mask] * grad[mask]) / (previous_m[mask].abs() * grad[mask].abs() + self.epsilon0)
                cos_sim = torch.clamp(cos_sim, min=0.0)
                psi = self.epsilon_prime + (1.0 - self.epsilon_prime) * cos_sim
                
                # Modulate the first moment with psi.
                modulated_exp_avg = exp_avg * psi
                
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                
                # Compute the update.
                update = -step_size * modulated_exp_avg / denom
                
                # Optional: Clip the update to stabilize training at high learning rates.
                max_update_norm = group.get('max_update_norm', None)
                if max_update_norm is not None:
                    update_norm = update.norm()
                    if update_norm > max_update_norm:
                        update = update * (max_update_norm / update_norm)
                
                # Apply the update.
                p.data.add_(update)
                
                # Update the stored previous momentum.
                state['previous_m'] = exp_avg.clone()
        
        return loss

import math
import torch
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F

class COSMIC(Optimizer):
    r"""Implements the proposed optimizer that enhances diffGrad by modulating the momentum 
    update with a cosine similarity based factor between the bias-corrected momentum and 
    the current gradient.
    
    This optimizer builds upon Adam/diffGrad and adjusts the effective step size according 
    to the directional consistency between the momentum and the gradient. When the gradient 
    direction is consistent with the momentum, the cosine similarity factor is high, thus 
    allowing larger updates; when they are misaligned, the update is dampened.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing running averages 
            of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
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

    def __setstate__(self, state):
        super(Experimental, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # Iterate over parameter groups.
        for group in self.param_groups:
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

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                step = state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                # Update biased first and second moment estimates.
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute bias-corrected first moment.
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                m_hat = exp_avg / bias_correction1

                # Compute denominator.
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                # Compute cosine similarity between m_hat and the current gradient.
                # Flatten the tensors to compute a single cosine similarity value per parameter tensor.
                m_flat = m_hat.view(1, -1)
                g_flat = grad.view(1, -1)
                cos_sim = F.cosine_similarity(m_flat, g_flat, dim=1, eps=group['eps'])
                # Remap cosine similarity from [-1, 1] to [0, 1]
                phi = 0.5 * (1.0 + cos_sim)
                # phi is a tensor with one element; extract its scalar value.
                phi = phi.item()

                # Compute step size.
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # Update parameters with the modulated momentum.
                p.data.addcdiv_(m_hat.mul(phi), denom, value=-step_size)

        return loss

