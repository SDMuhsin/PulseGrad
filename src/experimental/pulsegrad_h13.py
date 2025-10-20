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


class PulseGradH13(Optimizer):
    r"""H13: Adaptive Beta2 - Time-varying variance decay rate.

    Hypothesis: Variance estimation needs different decay rates across training phases.
    Early training (rapid changes): beta2=0.99 → Faster variance adaptation
    Late training (near convergence): beta2=0.9999 → More stable variance estimates

    Key modification:
        beta2_t = beta2_final - (beta2_final - beta2_init) * exp(-decay_rate * step)
        exp_avg_sq = beta2_t * exp_avg_sq + (1 - beta2_t) * new_val

    This is different from H6-H11 (boost) and H12 (adaptive gamma):
    - H6-H11 modified NUMERATOR (momentum amplification) → All failed
    - H12 modified DENOMINATOR-gamma (pulse penalty) → Failed (pulse penalty negligible)
    - H13 modifies DENOMINATOR-beta2 (variance decay rate) → Affects FULL variance

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): learning rate (default: 1e-3).
        beta1 (float, optional): coefficient for first moment (default: 0.9).
        beta2_init (float, optional): initial second moment decay coefficient (default: 0.99).
        beta2_final (float, optional): final second moment decay coefficient (default: 0.9999).
        decay_rate (float, optional): rate of exponential change for beta2 (default: 0.001).
        gamma (float, optional): regularization strength for gradient difference term (default: 0.3).
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8).
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0).
    """
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2_init=0.99, beta2_final=0.9999,
                 decay_rate=0.001, gamma=0.3, eps=1e-8, weight_decay=0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("Invalid beta1 parameter: {}".format(beta1))
        if not 0.0 <= beta2_init < 1.0:
            raise ValueError("Invalid beta2_init: {}".format(beta2_init))
        if not 0.0 <= beta2_final < 1.0:
            raise ValueError("Invalid beta2_final: {}".format(beta2_final))
        if beta2_final <= beta2_init:
            raise ValueError("beta2_final must be > beta2_init (we increase from low to high)")

        defaults = dict(lr=lr, beta1=beta1, beta2_init=beta2_init, beta2_final=beta2_final,
                       decay_rate=decay_rate, gamma=gamma, eps=eps, weight_decay=weight_decay)
        super(PulseGradH13, self).__init__(params, defaults)

        # Initialize diagnostics log
        self.log_file = open('results/h13_diagnostics.log', 'w')
        self.log_file.write('step,beta2_t,variance_mean,grad_squared_mean,pulse_penalty_mean,update_magnitude_mean,diff_mean\n')
        self.log_file.flush()

    def __setstate__(self, state):
        super(PulseGradH13, self).__setstate__(state)

    def __del__(self):
        if hasattr(self, 'log_file'):
            self.log_file.close()

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

        # Diagnostics accumulators
        diagnostics = {
            'beta2_t': 0.0,
            'variance': [],
            'grad_squared': [],
            'pulse_penalty': [],
            'update_magnitude': [],
            'diff': []
        }

        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            beta2_init = group['beta2_init']
            beta2_final = group['beta2_final']
            decay_rate = group['decay_rate']
            gamma = group['gamma']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('PulseGradH13 does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if not state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['previous_grad'] = torch.zeros_like(p)

                state['step'] += 1
                step = state['step']

                # ===== H13 KEY MODIFICATION: Compute adaptive beta2 =====
                # beta2_t increases from beta2_init (0.99) to beta2_final (0.9999) over training
                beta2_t = beta2_final - (beta2_final - beta2_init) * torch.exp(torch.tensor(-decay_rate * step))
                beta2_t_value = beta2_t.item()
                # ========================================================

                # Note: We don't use CUDA kernel because it doesn't support adaptive beta2
                # Pure Python path only
                exp_avg, exp_avg_sq, previous_grad = state['exp_avg'], state['exp_avg_sq'], state['previous_grad']

                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Update biased first moment estimate.
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Compute the absolute difference (THE PULSE).
                diff = torch.abs(previous_grad - grad)

                # Update biased second moment estimate with pulse
                pulse_penalty = gamma * diff.pow(2)
                grad_squared = grad.pow(2)
                new_val = grad_squared + pulse_penalty

                # ===== H13 KEY MODIFICATION: Use adaptive beta2_t =====
                # Update variance with time-varying beta2
                exp_avg_sq.mul_(beta2_t_value).add_(new_val, alpha=1 - beta2_t_value)
                # ======================================================

                # ===== H13 KEY MODIFICATION: Bias correction with adaptive beta2 =====
                # Note: This is approximate for time-varying beta2, but asymptotically correct
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2_t_value ** step
                # ====================================================================

                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2

                # Compute diffGrad friction coefficient (dfc)
                dfc = 1.0 / (1.0 + torch.exp(-diff))

                # Compute update
                momentum_total = m_hat * dfc
                denom = v_hat.sqrt().add_(eps)
                update = momentum_total / denom

                # Parameter update.
                p.add_(update, alpha=-lr)

                # Store the current gradient for the next step.
                state['previous_grad'].copy_(grad)

                # Collect diagnostics
                diagnostics['beta2_t'] = beta2_t_value
                diagnostics['variance'].append(exp_avg_sq.mean().item())
                diagnostics['grad_squared'].append(grad_squared.mean().item())
                diagnostics['pulse_penalty'].append(pulse_penalty.mean().item())
                diagnostics['update_magnitude'].append(update.abs().mean().item())
                diagnostics['diff'].append(diff.mean().item())

        # Log diagnostics every 50 steps
        if len(diagnostics['variance']) > 0:
            step = state['step']
            if step % 50 == 0 or step == 1:
                beta2_t = diagnostics['beta2_t']
                variance_mean = sum(diagnostics['variance']) / len(diagnostics['variance'])
                grad_squared_mean = sum(diagnostics['grad_squared']) / len(diagnostics['grad_squared'])
                pulse_penalty_mean = sum(diagnostics['pulse_penalty']) / len(diagnostics['pulse_penalty'])
                update_magnitude_mean = sum(diagnostics['update_magnitude']) / len(diagnostics['update_magnitude'])
                diff_mean = sum(diagnostics['diff']) / len(diagnostics['diff'])

                self.log_file.write(f'{step},{beta2_t:.6f},{variance_mean:.6f},{grad_squared_mean:.6f},'
                                  f'{pulse_penalty_mean:.6f},{update_magnitude_mean:.6f},{diff_mean:.6f}\n')
                self.log_file.flush()

        return loss
