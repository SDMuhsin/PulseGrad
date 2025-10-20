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


class PulseGradH2(Optimizer):
    r"""PulseGrad with smoothed pulse via EMA (H2).

    CHANGE: Use exponential moving average of diff instead of raw diff
    pulse_ema = beta_pulse * pulse_ema + (1-beta_pulse) * diff
    This reduces noise and captures genuine trajectory changes.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): learning rate (default: 1e-3).
        betas (Tuple[float, float], optional): coefficients used for computing running averages of
            gradient and its square (default: (0.9, 0.999)).
        gamma (float, optional): regularization strength for the gradient difference term (default: 0.3).
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8).
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0).
        beta_pulse (float, optional): EMA coefficient for pulse smoothing (default: 0.9).
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), gamma=0.3, eps=1e-8, weight_decay=0, beta_pulse=0.9):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= beta_pulse < 1.0:
            raise ValueError("Invalid beta_pulse parameter: {}".format(beta_pulse))

        defaults = dict(lr=lr, betas=betas, gamma=gamma, eps=eps, weight_decay=weight_decay, beta_pulse=beta_pulse)
        super(PulseGradH2, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PulseGradH2, self).__setstate__(state)

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
            beta_pulse = group['beta_pulse']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('PulseGradH2 does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if not state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['previous_grad'] = torch.zeros_like(p)
                    state['pulse_ema'] = torch.zeros_like(p)  # NEW: smoothed pulse

                state['step'] += 1

                # --- PURE PYTHON PATH (no CUDA kernel for now) ---
                exp_avg, exp_avg_sq, previous_grad = state['exp_avg'], state['exp_avg_sq'], state['previous_grad']
                pulse_ema = state['pulse_ema']
                step = state['step']

                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Update biased first moment estimate.
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Compute instantaneous pulse
                diff = torch.abs(previous_grad - grad)

                # CHANGE: Smooth the pulse with EMA
                pulse_ema.mul_(beta_pulse).add_(diff, alpha=1 - beta_pulse)

                # Update biased second moment estimate using SMOOTHED pulse
                new_val = grad.pow(2) + gamma * pulse_ema.pow(2)
                exp_avg_sq.mul_(beta2).add_(new_val, alpha=1 - beta2)

                # Bias correction.
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2

                # Compute friction coefficient using smoothed pulse
                dfc = 1.0 / (1.0 + torch.exp(-pulse_ema))

                # Parameter update.
                denom = v_hat.sqrt().add_(eps)
                p.addcdiv_(m_hat * dfc, denom, value=-lr)

                # Store the current gradient for the next step.
                state['previous_grad'].copy_(grad)

        return loss
