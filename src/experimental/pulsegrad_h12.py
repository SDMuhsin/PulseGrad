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


class PulseGradH12(Optimizer):
    r"""H12: Adaptive Gamma - Decreasing pulse penalty over training.

    Hypothesis: Variance decreases over training (diff: 0.0001 → 0.00001 observed in diagnostics).
    Therefore, pulse penalty should decrease proportionally.

    Early training (high variance): gamma=0.5 → Strong stabilization
    Late training (low variance): gamma=0.15 → Fast convergence

    Key modification:
        gamma_t = gamma_final + (gamma_init - gamma_final) * exp(-decay_rate * step)
        new_val = grad^2 + gamma_t * diff^2  (instead of constant gamma)

    This is fundamentally different from H6-H11 (boost attempts):
    - H6-H11 modified NUMERATOR (momentum amplification)
    - H12 modifies DENOMINATOR (variance penalty)

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): learning rate (default: 1e-3).
        betas (Tuple[float, float], optional): coefficients used for computing running averages of
            gradient and its square (default: (0.9, 0.999)).
        gamma_init (float, optional): initial pulse penalty strength (default: 0.5).
        gamma_final (float, optional): final pulse penalty strength (default: 0.15).
        decay_rate (float, optional): rate of exponential decay for gamma (default: 0.001).
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8).
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0).
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), gamma_init=0.5, gamma_final=0.15,
                 decay_rate=0.001, eps=1e-8, weight_decay=0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if gamma_init < 0.0:
            raise ValueError("Invalid gamma_init: {}".format(gamma_init))
        if gamma_final < 0.0:
            raise ValueError("Invalid gamma_final: {}".format(gamma_final))
        if gamma_init < gamma_final:
            raise ValueError("gamma_init must be >= gamma_final (we decay from high to low)")

        defaults = dict(lr=lr, betas=betas, gamma_init=gamma_init, gamma_final=gamma_final,
                       decay_rate=decay_rate, eps=eps, weight_decay=weight_decay)
        super(PulseGradH12, self).__init__(params, defaults)

        # Initialize diagnostics log
        self.log_file = open('results/h12_diagnostics.log', 'w')
        self.log_file.write('step,gamma_t,pulse_penalty_mean,grad_squared_mean,variance_mean,update_magnitude_mean,diff_mean\n')
        self.log_file.flush()

    def __setstate__(self, state):
        super(PulseGradH12, self).__setstate__(state)

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
            'gamma_t': 0.0,
            'pulse_penalty': [],
            'grad_squared': [],
            'variance': [],
            'update_magnitude': [],
            'diff': []
        }

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            gamma_init = group['gamma_init']
            gamma_final = group['gamma_final']
            decay_rate = group['decay_rate']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('PulseGradH12 does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if not state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['previous_grad'] = torch.zeros_like(p)

                state['step'] += 1
                step = state['step']

                # ===== H12 KEY MODIFICATION: Compute adaptive gamma =====
                # gamma_t decreases from gamma_init (0.5) to gamma_final (0.15) over training
                gamma_t = gamma_final + (gamma_init - gamma_final) * torch.exp(torch.tensor(-decay_rate * step))
                gamma_t_value = gamma_t.item()
                # ========================================================

                # Note: We don't use CUDA kernel because it doesn't support adaptive gamma
                # Pure Python path only
                exp_avg, exp_avg_sq, previous_grad = state['exp_avg'], state['exp_avg_sq'], state['previous_grad']

                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Update biased first moment estimate.
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Compute the absolute difference (THE PULSE).
                diff = torch.abs(previous_grad - grad)

                # ===== H12 KEY MODIFICATION: Use adaptive gamma_t =====
                # Update biased second moment estimate with time-varying gamma
                pulse_penalty = gamma_t_value * diff.pow(2)
                grad_squared = grad.pow(2)
                new_val = grad_squared + pulse_penalty
                exp_avg_sq.mul_(beta2).add_(new_val, alpha=1 - beta2)
                # ======================================================

                # Bias correction.
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

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
                diagnostics['gamma_t'] = gamma_t_value
                diagnostics['pulse_penalty'].append(pulse_penalty.mean().item())
                diagnostics['grad_squared'].append(grad_squared.mean().item())
                diagnostics['variance'].append(new_val.mean().item())
                diagnostics['update_magnitude'].append(update.abs().mean().item())
                diagnostics['diff'].append(diff.mean().item())

        # Log diagnostics every 50 steps
        if len(diagnostics['pulse_penalty']) > 0:
            step = state['step']
            if step % 50 == 0 or step == 1:
                gamma_t = diagnostics['gamma_t']
                pulse_penalty_mean = sum(diagnostics['pulse_penalty']) / len(diagnostics['pulse_penalty'])
                grad_squared_mean = sum(diagnostics['grad_squared']) / len(diagnostics['grad_squared'])
                variance_mean = sum(diagnostics['variance']) / len(diagnostics['variance'])
                update_magnitude_mean = sum(diagnostics['update_magnitude']) / len(diagnostics['update_magnitude'])
                diff_mean = sum(diagnostics['diff']) / len(diagnostics['diff'])

                self.log_file.write(f'{step},{gamma_t:.6f},{pulse_penalty_mean:.6f},{grad_squared_mean:.6f},'
                                  f'{variance_mean:.6f},{update_magnitude_mean:.6f},{diff_mean:.6f}\n')
                self.log_file.flush()

        return loss
