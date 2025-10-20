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


class PulseGradH9(Optimizer):
    r"""PulseGrad with Polynomial Safe Zone Boost - Corrected Calibration (H9).

    KEY FIX from H8: Uses LINEAR diff, not diff².

    H8 failed because diff² made values 10,000x smaller:
    - H8: 1/(1 + 10000*diff²) at diff=0.0001 → 1/(1+0.0001) ≈ 0.9999 ≈ 1.0 (no effect)
    - H9: 1/(1 + 10000*diff) at diff=0.0001 → 1/(1+1) = 0.5 (strong adaptive boost)

    Polynomial boost with LINEAR diff = 1 / (1 + k*diff)
    - More robust to diff scale than exponential
    - Correctly calibrated for observed diff~0.0001 scale
    - At diff=0.0001: boost ≈ 0.5 (50% boost)
    - At diff=0.001: boost ≈ 0.09 (9% boost)
    - At diff=0.01: boost ≈ 0.01 (1% boost)

    Mathematical formulation:
        boost_factor = 1 / (1 + k*diff)              # LINEAR diff, not diff²!
        boost_momentum = beta * m_hat * boost_factor
        momentum_total = m_hat * dfc + boost_momentum
        v = grad² + gamma * diff²                     # Unchanged (learned from H7)
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
        super(PulseGradH9, self).__init__(params, defaults)

        # Logging setup
        self.step_count = 0
        self.log_interval = 50
        self.log_file = './results/h9_diagnostics.log'

        # Create results dir if it doesn't exist
        os.makedirs('./results', exist_ok=True)

        # Initialize log file with headers
        with open(self.log_file, 'w') as f:
            f.write("step,diff_mean,diff_std,boost_factor_mean,boost_factor_std,boost_momentum_mean,momentum_ratio_mean,update_magnitude_mean\n")

    def __setstate__(self, state):
        super(PulseGradH9, self).__setstate__(state)

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
        all_boost_momentums = []
        all_momentum_ratios = []
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
                    raise RuntimeError('PulseGradH9 does not support sparse gradients')

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
                # This is critical! H7 failed by modifying this.
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

                # === H9 POLYNOMIAL SAFE ZONE BOOST (CORRECTED FROM H8) ===

                # CRITICAL FIX: Use LINEAR diff, not diff²!
                # H8 used: boost_factor = 1.0 / (1.0 + k * diff_squared)  ← WRONG
                # H9 uses: boost_factor = 1.0 / (1.0 + k * diff)           ← CORRECT
                #
                # At diff=0.0001 with k=10000:
                # - H8: 1/(1 + 10000*0.00000001) = 1/(1+0.0001) ≈ 0.9999 (no effect)
                # - H9: 1/(1 + 10000*0.0001) = 1/(1+1) = 0.5 (strong boost)
                boost_factor = 1.0 / (1.0 + k * diff)

                # Boost momentum when gradients are consistent
                boost_momentum = beta_boost * m_hat * boost_factor

                # Base momentum (standard PulseGrad)
                base_momentum = m_hat * dfc

                # Total momentum (ADDITIVE like H6, not replacing!)
                momentum_total = base_momentum + boost_momentum

                # Parameter update.
                denom = v_hat.sqrt().add_(eps)
                update = momentum_total / denom
                p.add_(update, alpha=-lr)

                # Collect logging data
                if should_log:
                    all_diffs.append(diff.detach())
                    all_boost_factors.append(boost_factor.detach())
                    all_boost_momentums.append(boost_momentum.abs().detach())
                    # Ratio of boost to base momentum (measures relative strength)
                    momentum_ratio = boost_momentum.abs() / (base_momentum.abs() + 1e-10)
                    all_momentum_ratios.append(momentum_ratio.detach())
                    all_updates.append(update.abs().detach())

                # Store the current gradient for the next step.
                state['previous_grad'].copy_(grad)

        # Log metrics
        if should_log and len(all_diffs) > 0:
            diff_cat = torch.cat([d.flatten() for d in all_diffs])
            boost_factor_cat = torch.cat([bf.flatten() for bf in all_boost_factors])
            boost_momentum_cat = torch.cat([bm.flatten() for bm in all_boost_momentums])
            momentum_ratio_cat = torch.cat([mr.flatten() for mr in all_momentum_ratios])
            update_cat = torch.cat([u.flatten() for u in all_updates])

            with open(self.log_file, 'a') as f:
                f.write(f"{self.step_count},"
                       f"{diff_cat.mean().item():.6f},"
                       f"{diff_cat.std().item():.6f},"
                       f"{boost_factor_cat.mean().item():.6f},"
                       f"{boost_factor_cat.std().item():.6f},"
                       f"{boost_momentum_cat.mean().item():.6f},"
                       f"{momentum_ratio_cat.mean().item():.6f},"
                       f"{update_cat.mean().item():.6f}\n")

            print(f"[H9 Diagnostics Step {self.step_count}] "
                  f"diff={diff_cat.mean().item():.6f}±{diff_cat.std().item():.6f}, "
                  f"boost_factor={boost_factor_cat.mean().item():.4f}±{boost_factor_cat.std().item():.4f}, "
                  f"boost_mom={boost_momentum_cat.mean().item():.6f}, "
                  f"mom_ratio={momentum_ratio_cat.mean().item():.3f}")

        return loss
