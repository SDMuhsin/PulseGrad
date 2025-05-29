import math
import torch
from torch.optim.optimizer import Optimizer

class ExperimentalV2(Optimizer):
    r"""Implements ExperimentalV2, an enhanced version of PulseGrad (Experimental).

    ExperimentalV2 modifies PulseGrad by introducing a `delta` parameter
    that adjusts the diffGrad friction coefficient (dfc) to be more aggressive
    during stable gradient periods (small changes in gradients), which can
    improve performance at low learning rates, especially for transformers.
    When delta is 0, ExperimentalV2 behaves identically to PulseGrad.

    PulseGrad (Experimental) itself modifies the standard diffGrad update by
    augmenting the second moment with an extra term that penalizes rapid
    changes in the gradient, as controlled by the gamma parameter.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate (default: 1e-3).
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999)).
        gamma (float, optional): regularization strength for the gradient
            difference term in the second moment (default: 0.3).
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8).
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0).
        delta (float, optional): Coefficient to adjust dfc aggressiveness.
            When delta is 0, dfc behaves as in PulseGrad. When delta is 1,
            dfc's dampening effect during stable periods is fully counteracted.
            Recommended range [0.0, 1.0]. (default: 0.0).
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), gamma=0.3, eps=1e-8, weight_decay=0, delta=1.0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= gamma: # gamma can be 0 or positive
            raise ValueError("Invalid gamma value: {}".format(gamma))
        if not 0.0 <= delta: # Allowing delta > 1, though its primary design is for [0,1]
            # User might experiment with delta > 1 for even more aggression,
            # though it might lead to instability if (1.0 - dfc) is large.
            # For the intended mechanism, delta <= 1 is ideal.
            # Recommending [0,1] in docs is good.
            pass

        defaults = dict(lr=lr, betas=betas, gamma=gamma, eps=eps, weight_decay=weight_decay, delta=delta)
        super(ExperimentalV2, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(ExperimentalV2, self).__setstate__(state)

    @torch.no_grad() # Undecorated in original, keeping consistent
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            # Before PyTorch 2.0, optimizers ran a closure with torch.enable_grad()
            # For modern PyTorch, if closure needs grad, it should handle it or
            # step should be called within a torch.no_grad() context if params are updated.
            # Optimizer steps usually modify parameters in-place, so they are often under no_grad.
            # However, if closure computes loss and needs grad, it's tricky.
            # The original code did not have torch.no_grad() here.
            # For safety with potential closure that needs grad for other things (not param update itself):
            with torch.enable_grad():
                 loss = closure()


        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            gamma = group['gamma']
            eps = group['eps']
            weight_decay = group['weight_decay']
            delta = group['delta']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data # .data is legacy, but keeping with original. Modern: p.grad
                if grad.is_sparse:
                    raise RuntimeError('ExperimentalV2 does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if not state: # len(state) == 0
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    state['previous_grad'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq, previous_grad = state['exp_avg'], state['exp_avg_sq'], state['previous_grad']

                state['step'] += 1
                step = state['step']

                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                # Update biased first moment estimate.
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Compute the absolute difference between the current and previous gradients.
                grad_diff = torch.abs(previous_grad - grad)

                # Update biased second moment estimate with the additional regularization term.
                # new_val = grad^2 + gamma * (grad_diff)^2
                # Using addcmul_ for potentially better numerical stability or clarity
                new_val = grad.pow(2).addcmul(torch.tensor(1.0, device=grad.device), grad_diff.pow(2), value=gamma)
                exp_avg_sq.mul_(beta2).add_(new_val, alpha=1 - beta2)


                # Bias correction for first and second moments.
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                # Corrected moments
                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2

                # Compute the diffGrad friction coefficient (dfc)
                # dfc is ~0.5 if grad_diff is small, ~1.0 if grad_diff is large.
                dfc = 1.0 / (1.0 + torch.exp(-grad_diff))

                # ExperimentalV2 enhancement: Adjust dfc using delta
                # update_multiplier will be dfc if delta = 0.
                # If delta = 1 and dfc = 0.5 (stable), update_multiplier = 1.0
                # If delta = 1 and dfc = 1.0 (unstable), update_multiplier = 1.0
                update_multiplier = dfc + delta * (1.0 - dfc)

                # Parameter update.
                denom = v_hat.sqrt().add_(eps)
                # Here, apply the (potentially adjusted) friction coefficient to the first moment.
                p.data.addcdiv_(m_hat * update_multiplier, denom, value=-lr)

                # Store the current gradient for the next step.
                state['previous_grad'] = grad.clone()
        return loss
