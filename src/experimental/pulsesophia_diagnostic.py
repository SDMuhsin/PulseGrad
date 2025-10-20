import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List
import os


class PulseSophiaDiagnostic(Optimizer):
    """
    Diagnostic version of PulseSophia that logs key metrics.

    Logs to file:
    - Gradient norm
    - Diff norm (gradient change)
    - Mean hessian term (rho * bs * h)
    - Mean pulse term
    - Pulse-to-hessian ratio
    - Mean total denominator
    - Update norm
    """

    def __init__(self, params, lr=1e-4, betas=(0.965, 0.99), rho=0.04,
                 gamma=0.3, weight_decay=1e-1, diagnostic_file=None, *, maximize: bool = False):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= rho:
            raise ValueError(f"Invalid rho parameter: {rho}")
        if not 0.0 <= gamma:
            raise ValueError(f"Invalid gamma parameter: {gamma}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, rho=rho, gamma=gamma,
                        weight_decay=weight_decay, maximize=maximize,
                        diagnostic_file=diagnostic_file)
        super(PulseSophiaDiagnostic, self).__init__(params, defaults)

        self.step_count = 0
        if diagnostic_file:
            os.makedirs(os.path.dirname(diagnostic_file) if os.path.dirname(diagnostic_file) else '.', exist_ok=True)
            with open(diagnostic_file, 'w') as f:
                f.write("step,grad_norm,diff_norm,hess_term_mean,pulse_term_mean,pulse_to_hess_ratio,denom_mean,update_norm\n")

    @torch.no_grad()
    def update_hessian(self):
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = torch.tensor(0.)
                    state['exp_avg'] = torch.zeros_like(p)
                    state['hessian'] = torch.zeros_like(p)
                    state['previous_grad'] = torch.zeros_like(p)

                if 'hessian' not in state:
                    state['hessian'] = torch.zeros_like(p)

                state['hessian'].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

    @torch.no_grad()
    def update_exp_avg(self):
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                state['exp_avg'].mul_(beta1).add_(p.grad, alpha=1 - beta1)

    @torch.no_grad()
    def step(self, closure=None, bs=5120):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.update_hessian()
        self.update_exp_avg()

        # Collect diagnostics
        all_grad_norms = []
        all_diff_norms = []
        all_hess_terms = []
        all_pulse_terms = []
        all_denoms = []
        all_update_norms = []

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            rho = group['rho']
            gamma = group['gamma']
            lr = group['lr']
            weight_decay = group['weight_decay']
            maximize = group['maximize']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad if not maximize else -p.grad

                if grad.is_sparse:
                    raise RuntimeError('PulseSophiaDiagnostic does not support sparse gradients')

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = torch.tensor(0.)
                    state['exp_avg'] = torch.zeros_like(p)
                    state['hessian'] = torch.zeros_like(p)
                    state['previous_grad'] = torch.zeros_like(p)

                if 'hessian' not in state:
                    state['hessian'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                hess = state['hessian']
                previous_grad = state['previous_grad']

                state['step'] += 1

                # Weight decay
                p.mul_(1 - lr * weight_decay)

                # Compute pulse
                diff = torch.abs(grad - previous_grad)
                pulse_term = gamma * diff.pow(2)

                # Sophia's adaptive ratio with pulse
                hess_term = rho * bs * hess
                denominator = hess_term + pulse_term + 1e-15
                ratio = (exp_avg.abs() / denominator).clamp(None, 1)

                # Update
                update = exp_avg.sign() * ratio

                # Collect diagnostics
                all_grad_norms.append(grad.norm().item())
                all_diff_norms.append(diff.norm().item())
                all_hess_terms.append(hess_term.mean().item())
                all_pulse_terms.append(pulse_term.mean().item())
                all_denoms.append(denominator.mean().item())
                all_update_norms.append(update.norm().item())

                # Parameter update
                p.addcmul_(exp_avg.sign(), ratio, value=-lr)

                # Store gradient
                state['previous_grad'].copy_(grad)

        # Write diagnostics
        diagnostic_file = self.param_groups[0].get('diagnostic_file')
        if diagnostic_file and len(all_grad_norms) > 0:
            import numpy as np
            grad_norm = np.mean(all_grad_norms)
            diff_norm = np.mean(all_diff_norms)
            hess_term_mean = np.mean(all_hess_terms)
            pulse_term_mean = np.mean(all_pulse_terms)
            denom_mean = np.mean(all_denoms)
            update_norm = np.mean(all_update_norms)
            pulse_to_hess = pulse_term_mean / hess_term_mean if hess_term_mean > 0 else 0

            with open(diagnostic_file, 'a') as f:
                f.write(f"{self.step_count},{grad_norm:.6f},{diff_norm:.6f},{hess_term_mean:.6f},{pulse_term_mean:.6f},{pulse_to_hess:.6f},{denom_mean:.6f},{update_norm:.6f}\n")

        self.step_count += 1
        return loss
