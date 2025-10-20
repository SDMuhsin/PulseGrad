import math
import torch
from torch.optim.optimizer import Optimizer
import os

# Import the compiled CUDA extension
try:
    import experimental_ext
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: CUDA extension for Experimental optimizer not found. Running in pure Python mode.")


class PulseGradH11(Optimizer):
    r"""PulseGrad with Multiplicative Constant Boost (H11) - FINAL BOOST TEST.

    This is our final test of boost mechanisms. Tests whether H6's success was due to
    CONSTANT (time-invariant) boost rather than calibration or luck.

    H6 (mult, constant ~2x): +0.0017% ✓
    H10 (mult, adaptive 1.35-1.50x): -0.0366% ❌

    The difference: H6's boost was CONSTANT, H10's was TIME-VARYING.

    H11 explicitly tests constant boost hypothesis:
    - momentum = m_hat * dfc * 2.0 (CONSTANT throughout training)
    - No adaptivity, no complexity, just uniform scaling

    If H11 matches H6: Constant boost is real → try stronger constants
    If H11 fails: H6 was noise → ABANDON boost mechanisms entirely

    Mathematical formulation:
        momentum_total = m_hat * dfc * constant_multiplier  # constant_multiplier = 2.0
        v = grad² + gamma * diff²                           # Unchanged
        update = -lr * momentum_total / sqrt(v + eps)

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): learning rate (default: 1e-3).
        betas (Tuple[float, float], optional): coefficients used for computing running averages of
            gradient and its square (default: (0.9, 0.999)).
        gamma (float, optional): regularization strength for the gradient difference term (default: 0.3).
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8).
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0).
        constant_multiplier (float, optional): constant boost multiplier (default: 2.0).
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), gamma=0.3, eps=1e-8, weight_decay=0, constant_multiplier=2.0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, betas=betas, gamma=gamma, eps=eps, weight_decay=weight_decay, constant_multiplier=constant_multiplier)
        super(PulseGradH11, self).__init__(params, defaults)

        # Minimal logging (just verify constant is applied)
        self.step_count = 0
        self.log_interval = 500  # Less frequent since nothing should vary
        self.log_file = './results/h11_diagnostics.log'

        # Create results dir if it doesn't exist
        os.makedirs('./results', exist_ok=True)

        # Initialize log file with headers
        with open(self.log_file, 'w') as f:
            f.write("step,constant_multiplier,effective_momentum_mean,update_magnitude_mean\n")

    def __setstate__(self, state):
        super(PulseGradH11, self).__setstate__(state)

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

        self.step_count += 1
        should_log = (self.step_count % self.log_interval == 0)

        # Accumulators for logging
        all_effective_momentums = []
        all_updates = []

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            gamma = group['gamma']
            eps = group['eps']
            weight_decay = group['weight_decay']
            constant_multiplier = group['constant_multiplier']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('PulseGradH11 does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if not state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['previous_grad'] = torch.zeros_like(p)

                state['step'] += 1

                exp_avg, exp_avg_sq, previous_grad = state['exp_avg'], state['exp_avg_sq'], state['previous_grad']
                step = state['step']

                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Update biased first moment estimate.
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Compute the absolute difference (THE PULSE).
                diff = torch.abs(previous_grad - grad)

                # Update biased second moment estimate (UNCHANGED - keeps stabilization).
                diff_squared = diff.pow(2)
                new_val = grad.pow(2) + gamma * diff_squared
                exp_avg_sq.mul_(beta2).add_(new_val, alpha=1 - beta2)

                # Bias correction.
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2

                # Compute diffGrad friction coefficient (UNCHANGED)
                dfc = 1.0 / (1.0 + torch.exp(-diff))

                # === H11 CONSTANT BOOST ===
                # CRITICAL: This is CONSTANT (not adaptive like H10)
                # Just multiply momentum by constant factor
                momentum_total = m_hat * dfc * constant_multiplier

                # Parameter update.
                denom = v_hat.sqrt().add_(eps)
                update = momentum_total / denom
                p.add_(update, alpha=-lr)

                # Collect logging data
                if should_log:
                    all_effective_momentums.append((m_hat * dfc * constant_multiplier).abs().detach())
                    all_updates.append(update.abs().detach())

                # Store the current gradient for the next step.
                state['previous_grad'].copy_(grad)

        # Log metrics
        if should_log and len(all_effective_momentums) > 0:
            effective_momentum_cat = torch.cat([em.flatten() for em in all_effective_momentums])
            update_cat = torch.cat([u.flatten() for u in all_updates])

            with open(self.log_file, 'a') as f:
                f.write(f"{self.step_count},"
                       f"{constant_multiplier:.6f},"
                       f"{effective_momentum_cat.mean().item():.6f},"
                       f"{update_cat.mean().item():.6f}\n")

            print(f"[H11 Diagnostics Step {self.step_count}] "
                  f"constant_mult={constant_multiplier:.1f}, "
                  f"eff_mom={effective_momentum_cat.mean().item():.6f}, "
                  f"update_mag={update_cat.mean().item():.6f}")

        return loss
