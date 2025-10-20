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


class PulseGradH14(Optimizer):
    r"""H14: Linear Pulse - Core mechanism redesign.

    Hypothesis: Linear pulse scaling (gamma * diff instead of gamma * diff²) maintains
    pulse penalty relevance when gradients are small.

    H12 discovered: pulse_penalty ≈ 0 for 95% of training because diff² scaling makes
    it negligible when diff < 0.001.

    H14 solution: Use linear scaling to maintain relevance throughout training.

    Key modification:
        # Baseline: pulse_penalty = gamma * diff²  (quadratic, gamma=0.3)
        # H14: pulse_penalty = gamma * diff  (linear, gamma=0.0001)

    When diff = 0.001:
        Baseline: 0.3 * 0.000001 = 0.0000003 (negligible)
        H14: 0.0001 * 0.001 = 0.0000001 (1000x larger relative magnitude)

    This is our FIRST CORE MECHANISM REDESIGN, not just a parameter tweak.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): learning rate (default: 1e-3).
        betas (Tuple[float, float], optional): coefficients used for computing running averages (default: (0.9, 0.999)).
        gamma (float, optional): pulse penalty strength for LINEAR diff (default: 0.0001).
        eps (float, optional): term added to the denominator (default: 1e-8).
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0).
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), gamma=0.0001, eps=1e-8, weight_decay=0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if gamma < 0.0:
            raise ValueError("Invalid gamma: {}".format(gamma))

        defaults = dict(lr=lr, betas=betas, gamma=gamma, eps=eps, weight_decay=weight_decay)
        super(PulseGradH14, self).__init__(params, defaults)

        # Initialize diagnostics log
        self.log_file = open('results/h14_diagnostics.log', 'w')
        self.log_file.write('step,pulse_penalty_mean,grad_squared_mean,pulse_to_grad_ratio,variance_mean,update_magnitude_mean,diff_mean\n')
        self.log_file.flush()

    def __setstate__(self, state):
        super(PulseGradH14, self).__setstate__(state)

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
            'pulse_penalty': [],
            'grad_squared': [],
            'pulse_to_grad_ratio': [],
            'variance': [],
            'update_magnitude': [],
            'diff': []
        }

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
                    raise RuntimeError('PulseGradH14 does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if not state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['previous_grad'] = torch.zeros_like(p)

                state['step'] += 1
                step = state['step']

                # Note: We don't use CUDA kernel because it doesn't support linear pulse
                # Pure Python path only
                exp_avg, exp_avg_sq, previous_grad = state['exp_avg'], state['exp_avg_sq'], state['previous_grad']

                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Update biased first moment estimate.
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Compute the absolute difference (THE PULSE).
                diff = torch.abs(previous_grad - grad)

                # ===== H14 KEY MODIFICATION: LINEAR pulse penalty =====
                # Baseline: pulse_penalty = gamma * diff²  (quadratic, gamma=0.3)
                # H14: pulse_penalty = gamma * diff  (linear, gamma=0.0001)
                pulse_penalty = gamma * diff  # LINEAR, not diff.pow(2)
                grad_squared = grad.pow(2)
                new_val = grad_squared + pulse_penalty
                # ======================================================

                # Update biased second moment estimate
                exp_avg_sq.mul_(beta2).add_(new_val, alpha=1 - beta2)

                # Bias correction
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

                # Parameter update
                p.add_(update, alpha=-lr)

                # Store the current gradient for the next step
                state['previous_grad'].copy_(grad)

                # Collect diagnostics
                pulse_penalty_mean = pulse_penalty.mean().item()
                grad_squared_mean = grad_squared.mean().item()

                # Compute pulse-to-grad ratio (key metric)
                # Avoid division by zero
                pulse_to_grad_ratio = (pulse_penalty_mean / grad_squared_mean) if grad_squared_mean > 1e-10 else 0.0

                diagnostics['pulse_penalty'].append(pulse_penalty_mean)
                diagnostics['grad_squared'].append(grad_squared_mean)
                diagnostics['pulse_to_grad_ratio'].append(pulse_to_grad_ratio)
                diagnostics['variance'].append(new_val.mean().item())
                diagnostics['update_magnitude'].append(update.abs().mean().item())
                diagnostics['diff'].append(diff.mean().item())

        # Log diagnostics every 50 steps
        if len(diagnostics['pulse_penalty']) > 0:
            step = state['step']
            if step % 50 == 0 or step == 1:
                pulse_penalty_mean = sum(diagnostics['pulse_penalty']) / len(diagnostics['pulse_penalty'])
                grad_squared_mean = sum(diagnostics['grad_squared']) / len(diagnostics['grad_squared'])
                pulse_to_grad_ratio = sum(diagnostics['pulse_to_grad_ratio']) / len(diagnostics['pulse_to_grad_ratio'])
                variance_mean = sum(diagnostics['variance']) / len(diagnostics['variance'])
                update_magnitude_mean = sum(diagnostics['update_magnitude']) / len(diagnostics['update_magnitude'])
                diff_mean = sum(diagnostics['diff']) / len(diagnostics['diff'])

                self.log_file.write(f'{step},{pulse_penalty_mean:.6f},{grad_squared_mean:.6f},'
                                  f'{pulse_to_grad_ratio:.6f},{variance_mean:.6f},'
                                  f'{update_magnitude_mean:.6f},{diff_mean:.6f}\n')
                self.log_file.flush()

        return loss
