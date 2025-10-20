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


class PulseGradDual(Optimizer):
    r"""PulseGrad with Dual Safe Zone Acceleration (H7).

    KEY INNOVATION: Simultaneously accelerates in safe zones via:
    1. NUMERATOR: Additive momentum boost when diff is small
    2. DENOMINATOR: Adaptive gamma that reduces pulse penalty when diff is small

    Mathematical formulation:
        safety = 1 - exp(-alpha*diff)          # 0 when safe, 1 when risky
        boost_momentum = beta*m_hat * exp(-alpha*diff)  # Large when safe
        gamma_adaptive = gamma * safety         # Small when safe

        momentum_total = m_hat * dfc + boost_momentum
        v = grad² + gamma_adaptive * diff²
        update = -lr * momentum_total / sqrt(v + eps)

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): learning rate (default: 1e-3).
        betas (Tuple[float, float], optional): coefficients used for computing running averages of
            gradient and its square (default: (0.9, 0.999)).
        gamma (float, optional): regularization strength for the gradient difference term (default: 0.3).
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8).
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0).
        alpha (float, optional): safe zone detection rate (default: 3.0).
        beta (float, optional): safe zone boost strength (default: 0.5).
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), gamma=0.3, eps=1e-8, weight_decay=0, alpha=3.0, beta=0.5):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, betas=betas, gamma=gamma, eps=eps, weight_decay=weight_decay, alpha=alpha, beta=beta)
        super(PulseGradDual, self).__init__(params, defaults)

        # Logging setup
        self.step_count = 0
        self.log_interval = 50
        self.log_file = './results/h7_diagnostics.log'

        # Create results dir if it doesn't exist
        os.makedirs('./results', exist_ok=True)

        # Initialize log file
        with open(self.log_file, 'w') as f:
            f.write("step,diff_mean,diff_std,safety_mean,boost_mean,gamma_adaptive_mean,update_magnitude_mean\n")

    def __setstate__(self, state):
        super(PulseGradDual, self).__setstate__(state)

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
        all_safeties = []
        all_boosts = []
        all_gammas_adaptive = []
        all_updates = []

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            gamma = group['gamma']
            eps = group['eps']
            weight_decay = group['weight_decay']
            alpha = group['alpha']
            beta_boost = group['beta']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('PulseGradDual does not support sparse gradients')

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

                # === H7 DUAL ACCELERATION ===

                # Compute exp(-alpha*diff) once for efficiency
                exp_neg_alpha_diff = torch.exp(-alpha * diff)

                # 1. Safety metric: 0 when safe (diff≈0), 1 when risky (diff>0.5)
                safety = 1.0 - exp_neg_alpha_diff

                # 2. Adaptive gamma: reduces pulse penalty in safe zones
                gamma_adaptive = gamma * safety

                # Update biased second moment with ADAPTIVE gamma
                new_val = grad.pow(2) + gamma_adaptive * diff.pow(2)
                exp_avg_sq.mul_(beta2).add_(new_val, alpha=1 - beta2)

                # Bias correction.
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2

                # Compute diffGrad friction coefficient (UNCHANGED)
                dfc = 1.0 / (1.0 + torch.exp(-diff))

                # 3. Safe zone momentum boost: adds extra momentum when diff is small
                boost_momentum = beta_boost * m_hat * exp_neg_alpha_diff

                # 4. Total momentum: base + boost
                momentum_total = m_hat * dfc + boost_momentum

                # Parameter update with dual acceleration
                denom = v_hat.sqrt().add_(eps)
                update = momentum_total / denom
                p.add_(update, alpha=-lr)

                # Collect logging data
                if should_log:
                    all_diffs.append(diff.detach())
                    all_safeties.append(safety.detach())
                    all_boosts.append(boost_momentum.abs().detach())
                    all_gammas_adaptive.append(gamma_adaptive.detach())
                    all_updates.append(update.abs().detach())

                # Store the current gradient for the next step.
                state['previous_grad'].copy_(grad)

        # Log metrics
        if should_log and len(all_diffs) > 0:
            diff_cat = torch.cat([d.flatten() for d in all_diffs])
            safety_cat = torch.cat([s.flatten() for s in all_safeties])
            boost_cat = torch.cat([b.flatten() for b in all_boosts])
            gamma_cat = torch.cat([g.flatten() for g in all_gammas_adaptive])
            update_cat = torch.cat([u.flatten() for u in all_updates])

            with open(self.log_file, 'a') as f:
                f.write(f"{self.step_count},"
                       f"{diff_cat.mean().item():.6f},"
                       f"{diff_cat.std().item():.6f},"
                       f"{safety_cat.mean().item():.6f},"
                       f"{boost_cat.mean().item():.6f},"
                       f"{gamma_cat.mean().item():.6f},"
                       f"{update_cat.mean().item():.6f}\n")

            print(f"[H7 Diagnostics Step {self.step_count}] "
                  f"diff={diff_cat.mean().item():.4f}±{diff_cat.std().item():.4f}, "
                  f"safety={safety_cat.mean().item():.3f}, "
                  f"boost={boost_cat.mean().item():.6f}, "
                  f"gamma_eff={gamma_cat.mean().item():.4f}")

        return loss
