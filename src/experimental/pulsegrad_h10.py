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


class PulseGradH10(Optimizer):
    r"""PulseGrad with Multiplicative Polynomial Boost (H10).

    KEY DIFFERENCE from H9: MULTIPLICATIVE boost instead of ADDITIVE.

    H9 (additive) failed despite correct calibration (-0.0516%):
    - momentum = m_hat * dfc + beta * m_hat * boost_factor
    - Disrupted PulseGrad's balance between safe and risky zones

    H6 (multiplicative) succeeded despite miscalibration (+0.0017%):
    - momentum = m_hat * dfc * (1 + boost)
    - Preserved relative scaling between zones

    H10 combines H6's mechanism with H9's calibration:
    - momentum = m_hat * dfc * (1 + beta * boost_factor)
    - Preserves balance while providing adaptive boost

    Mathematical formulation:
        boost_factor = 1 / (1 + k*diff)                          # H9's correct calibration
        momentum_total = m_hat * dfc * (1.0 + beta * boost_factor)  # H6's mechanism
        v = grad² + gamma * diff²                                 # Unchanged
        update = -lr * momentum_total / sqrt(v + eps)

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): learning rate (default: 1e-3).
        betas (Tuple[float, float], optional): coefficients used for computing running averages of
            gradient and its square (default: (0.9, 0.999)).
        gamma (float, optional): regularization strength for the gradient difference term (default: 0.3).
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8).
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0).
        k (float, optional): polynomial decay rate for safe zone detection (default: 10000.0).
        beta (float, optional): safe zone boost strength (default: 0.5).
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), gamma=0.3, eps=1e-8, weight_decay=0, k=10000.0, beta=0.5):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, betas=betas, gamma=gamma, eps=eps, weight_decay=weight_decay, k=k, beta=beta)
        super(PulseGradH10, self).__init__(params, defaults)

        # Logging setup
        self.step_count = 0
        self.log_interval = 50
        self.log_file = './results/h10_diagnostics.log'

        # Create results dir if it doesn't exist
        os.makedirs('./results', exist_ok=True)

        # Initialize log file with headers
        with open(self.log_file, 'w') as f:
            f.write("step,diff_mean,diff_std,boost_factor_mean,boost_factor_std,momentum_multiplier_mean,effective_momentum_mean,update_magnitude_mean\n")

    def __setstate__(self, state):
        super(PulseGradH10, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step with comprehensive logging.

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
        all_diffs = []
        all_boost_factors = []
        all_momentum_multipliers = []
        all_effective_momentums = []
        all_updates = []

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            gamma = group['gamma']
            eps = group['eps']
            weight_decay = group['weight_decay']
            k = group['k']
            beta_boost = group['beta']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('PulseGradH10 does not support sparse gradients')

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

                # === H10 MULTIPLICATIVE POLYNOMIAL BOOST ===

                # Polynomial boost factor with LINEAR diff (from H9)
                boost_factor = 1.0 / (1.0 + k * diff)

                # CRITICAL: MULTIPLICATIVE boost (like H6), NOT additive (like H9)
                # H6 formula: m_hat * dfc * (1 + boost)
                # H9 formula: m_hat * dfc + beta * m_hat * boost_factor  ← ADDITIVE (failed)
                # H10 formula: m_hat * dfc * (1 + beta * boost_factor)  ← MULTIPLICATIVE
                momentum_multiplier = 1.0 + beta_boost * boost_factor
                momentum_total = m_hat * dfc * momentum_multiplier

                # Parameter update.
                denom = v_hat.sqrt().add_(eps)
                update = momentum_total / denom
                p.add_(update, alpha=-lr)

                # Collect logging data
                if should_log:
                    all_diffs.append(diff.detach())
                    all_boost_factors.append(boost_factor.detach())
                    all_momentum_multipliers.append(momentum_multiplier.detach())
                    all_effective_momentums.append((m_hat * dfc * momentum_multiplier).abs().detach())
                    all_updates.append(update.abs().detach())

                # Store the current gradient for the next step.
                state['previous_grad'].copy_(grad)

        # Log metrics
        if should_log and len(all_diffs) > 0:
            diff_cat = torch.cat([d.flatten() for d in all_diffs])
            boost_factor_cat = torch.cat([bf.flatten() for bf in all_boost_factors])
            momentum_multiplier_cat = torch.cat([mm.flatten() for mm in all_momentum_multipliers])
            effective_momentum_cat = torch.cat([em.flatten() for em in all_effective_momentums])
            update_cat = torch.cat([u.flatten() for u in all_updates])

            with open(self.log_file, 'a') as f:
                f.write(f"{self.step_count},"
                       f"{diff_cat.mean().item():.6f},"
                       f"{diff_cat.std().item():.6f},"
                       f"{boost_factor_cat.mean().item():.6f},"
                       f"{boost_factor_cat.std().item():.6f},"
                       f"{momentum_multiplier_cat.mean().item():.6f},"
                       f"{effective_momentum_cat.mean().item():.6f},"
                       f"{update_cat.mean().item():.6f}\n")

            print(f"[H10 Diagnostics Step {self.step_count}] "
                  f"diff={diff_cat.mean().item():.6f}±{diff_cat.std().item():.6f}, "
                  f"boost_factor={boost_factor_cat.mean().item():.4f}±{boost_factor_cat.std().item():.4f}, "
                  f"multiplier={momentum_multiplier_cat.mean().item():.4f}, "
                  f"eff_mom={effective_momentum_cat.mean().item():.6f}")

        return loss
