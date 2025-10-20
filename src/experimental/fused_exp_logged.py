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


class ExperimentalLogged(Optimizer):
    r"""Logged version of Experimental optimizer for analysis."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), gamma=0.3, eps=1e-8, weight_decay=0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, betas=betas, gamma=gamma, eps=eps, weight_decay=weight_decay)
        super(ExperimentalLogged, self).__init__(params, defaults)

        # Logging state
        self.step_count = 0
        self.log_interval = 50
        self.metrics = {
            'diff_mean': [],
            'diff_std': [],
            'dfc_mean': [],
            'dfc_std': [],
            'v_base_mean': [],  # grad^2 component
            'v_pulse_mean': [],  # gamma*diff^2 component
            'step_size_mean': [],
        }

    def __setstate__(self, state):
        super(ExperimentalLogged, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step with logging."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.step_count += 1
        should_log = (self.step_count % self.log_interval == 0)

        # Accumulators for logging
        all_diffs = []
        all_dfcs = []
        all_v_base = []
        all_v_pulse = []
        all_step_sizes = []

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            gamma = group['gamma']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Experimental does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if not state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['previous_grad'] = torch.zeros_like(p)

                state['step'] += 1

                # --- PURE PYTHON PATH WITH LOGGING ---
                exp_avg, exp_avg_sq, previous_grad = state['exp_avg'], state['exp_avg_sq'], state['previous_grad']
                step = state['step']

                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Update biased first moment estimate.
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Compute the absolute difference (THE PULSE).
                diff = torch.abs(previous_grad - grad)

                # Update biased second moment estimate.
                v_base = grad.pow(2)
                v_pulse = gamma * diff.pow(2)
                new_val = v_base + v_pulse
                exp_avg_sq.mul_(beta2).add_(new_val, alpha=1 - beta2)

                # Bias correction.
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2

                # Compute diffGrad friction coefficient (dfc)
                dfc = 1.0 / (1.0 + torch.exp(-diff))

                # Parameter update.
                denom = v_hat.sqrt().add_(eps)
                step_size = (m_hat * dfc) / denom
                p.add_(step_size, alpha=-lr)

                # Collect logging data
                if should_log:
                    all_diffs.append(diff.detach())
                    all_dfcs.append(dfc.detach())
                    all_v_base.append(v_base.detach())
                    all_v_pulse.append(v_pulse.detach())
                    all_step_sizes.append(step_size.abs().detach())

                # Store the current gradient for the next step.
                state['previous_grad'].copy_(grad)

        # Log metrics
        if should_log and len(all_diffs) > 0:
            diff_cat = torch.cat([d.flatten() for d in all_diffs])
            dfc_cat = torch.cat([d.flatten() for d in all_dfcs])
            v_base_cat = torch.cat([d.flatten() for d in all_v_base])
            v_pulse_cat = torch.cat([d.flatten() for d in all_v_pulse])
            step_size_cat = torch.cat([d.flatten() for d in all_step_sizes])

            self.metrics['diff_mean'].append(diff_cat.mean().item())
            self.metrics['diff_std'].append(diff_cat.std().item())
            self.metrics['dfc_mean'].append(dfc_cat.mean().item())
            self.metrics['dfc_std'].append(dfc_cat.std().item())
            self.metrics['v_base_mean'].append(v_base_cat.mean().item())
            self.metrics['v_pulse_mean'].append(v_pulse_cat.mean().item())
            self.metrics['step_size_mean'].append(step_size_cat.mean().item())

            print(f"[Step {self.step_count}] diff={diff_cat.mean().item():.6f}±{diff_cat.std().item():.6f}, "
                  f"dfc={dfc_cat.mean().item():.4f}±{dfc_cat.std().item():.4f}, "
                  f"v_base={v_base_cat.mean().item():.6f}, v_pulse={v_pulse_cat.mean().item():.6f}, "
                  f"v_pulse/v_base={v_pulse_cat.mean().item()/max(v_base_cat.mean().item(),1e-10):.4f}")

        return loss
