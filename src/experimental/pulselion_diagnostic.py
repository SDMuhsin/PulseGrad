from __future__ import annotations
from typing import Tuple, Callable
import torch
from torch.optim.optimizer import Optimizer
import os


class PulseLionDiagnostic(Optimizer):
    """
    Diagnostic version of PulseLion that logs key metrics.

    Logs to file:
    - Gradient norm
    - Diff norm (gradient change)
    - Mean pulse term value
    - Mean denominator value
    - Update norm
    - Pulse-to-total ratio
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        gamma: float = 0.3,
        weight_decay: float = 0.0,
        diagnostic_file: str = None,
    ):
        assert lr > 0., f"Invalid learning rate: {lr}"
        assert all([0. <= beta <= 1. for beta in betas]), f"Invalid betas: {betas}"
        assert gamma >= 0., f"Invalid gamma: {gamma}"

        defaults = dict(
            lr=lr,
            betas=betas,
            gamma=gamma,
            weight_decay=weight_decay,
            diagnostic_file=diagnostic_file
        )

        super().__init__(params, defaults)

        self.step_count = 0
        if diagnostic_file:
            # Write header
            os.makedirs(os.path.dirname(diagnostic_file) if os.path.dirname(diagnostic_file) else '.', exist_ok=True)
            with open(diagnostic_file, 'w') as f:
                f.write("step,grad_norm,diff_norm,pulse_term_mean,denom_mean,update_norm,pulse_ratio\n")

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Collect diagnostics across all parameters
        all_grad_norms = []
        all_diff_norms = []
        all_pulse_terms = []
        all_denoms = []
        all_update_norms = []

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('PulseLionDiagnostic does not support sparse gradients')

                lr = group['lr']
                beta1, beta2 = group['betas']
                gamma = group['gamma']
                weight_decay = group['weight_decay']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['previous_grad'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                previous_grad = state['previous_grad']

                # Weight decay
                p.data.mul_(1. - lr * weight_decay)

                # Compute gradient difference (pulse signal)
                diff = torch.abs(grad - previous_grad)

                # Interpolated momentum
                m_interp = exp_avg.clone().mul_(beta1).add(grad, alpha=1. - beta1)

                # Sign-based update
                update_sign = m_interp.sign()

                # Pulse stabilization
                pulse_term = gamma * diff.pow(2)
                denom = (1.0 + pulse_term).sqrt()
                update = update_sign / denom

                # Collect diagnostics
                all_grad_norms.append(grad.norm().item())
                all_diff_norms.append(diff.norm().item())
                all_pulse_terms.append(pulse_term.mean().item())
                all_denoms.append(denom.mean().item())
                all_update_norms.append(update.norm().item())

                # Parameter update
                p.add_(update, alpha=-lr)

                # Update momentum
                exp_avg.mul_(beta2).add_(grad, alpha=1. - beta2)

                # Store current gradient
                state['previous_grad'].copy_(grad)

        # Write diagnostics to file
        diagnostic_file = self.param_groups[0].get('diagnostic_file')
        if diagnostic_file and len(all_grad_norms) > 0:
            import numpy as np
            grad_norm = np.mean(all_grad_norms)
            diff_norm = np.mean(all_diff_norms)
            pulse_term_mean = np.mean(all_pulse_terms)
            denom_mean = np.mean(all_denoms)
            update_norm = np.mean(all_update_norms)
            pulse_ratio = pulse_term_mean / denom_mean if denom_mean > 0 else 0

            with open(diagnostic_file, 'a') as f:
                f.write(f"{self.step_count},{grad_norm:.6f},{diff_norm:.6f},{pulse_term_mean:.6f},{denom_mean:.6f},{update_norm:.6f},{pulse_ratio:.6f}\n")

        self.step_count += 1
        return loss
