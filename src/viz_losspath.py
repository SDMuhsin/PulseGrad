# optimizer_trajectory_1d_comparison.py

import os
import warnings
warnings.filterwarnings("ignore", message="This overload of add_ is deprecated")

import torch
from experimental.diffgrad import diffgrad
from experimental.exp import Experimental as PulseGrad
import numpy as np
import matplotlib.pyplot as plt

# === FINAL 1D TOY FUNCTIONS FOR Δg-DRIVEN VARIANCE BENEFIT ===

def sharp_tanh_steps(x):
    """
    Quadratic base with two steep 'steps' via tanh:
    f(x) = 0.5 x² + 5 tanh(100(x−1)) − 5 tanh(100(x−2))
    Sharp gradient changes at x=1,2 maximize (Δg)² impact.
    """
    def torch_fn(t):
        return 0.5 * t**2 + 5 * torch.tanh(100 * (t - 1)) - 5 * torch.tanh(100 * (t - 2))
    def np_fn(t):
        return 0.5 * t**2 + 5 * np.tanh(100 * (t - 1)) - 5 * np.tanh(100 * (t - 2))
    return torch_fn(x) if isinstance(x, torch.Tensor) else np_fn(x)


def fractal_sine(x):
    """
    Multi-scale sine series: high vs. low freq.
    f(x) = Σ_{i=1}^{10} 2^{-i} sin(2^i x) + 0.02 x²
    Designed to produce large Δg between iterations.
    """
    def torch_fn(t):
        total = torch.zeros_like(t)
        for i in range(1, 11):
            total += (1 / (2 ** i)) * torch.sin((2 ** i) * t)
        return total + 0.02 * t**2
    def np_fn(t):
        total = np.zeros_like(t)
        for i in range(1, 11):
            total += (1 / (2 ** i)) * np.sin((2 ** i) * t)
        return total + 0.02 * t**2
    return torch_fn(x) if isinstance(x, torch.Tensor) else np_fn(x)


def tanh_sine_osc(x):
    """
    Quadratic plus sharp sinusoidal-induced transitions:
    f(x) = x² + 5 tanh(100 sin(5 x)) + 5 tanh(100 sin(7 x))
    Sinusoidal cores create abrupt Δg near zero crossings.
    """
    def torch_fn(t):
        return (
            t**2
            + 5 * torch.tanh(100 * torch.sin(5 * t))
            + 5 * torch.tanh(100 * torch.sin(7 * t))
        )
    def np_fn(t):
        return (
            t**2
            + 5 * np.tanh(100 * np.sin(5 * t))
            + 5 * np.tanh(100 * np.sin(7 * t))
        )
    return torch_fn(x) if isinstance(x, torch.Tensor) else np_fn(x)

# Bundle functions

toy_functions = {
    'sharp_tanh_steps': sharp_tanh_steps,
    'fractal_sine':     fractal_sine,
    'tanh_sine_osc':    tanh_sine_osc,
}

# LaTeX equations and display names for camera-ready presentation

equations = {
    'sharp_tanh_steps': r"$0.5 x^2 + 5\tanh(100(x-1)) - 5\tanh(100(x-2))$",
    'fractal_sine':     r"$\sum_{i=1}^{10}2^{-i}\sin(2^i x) + 0.02 x^2$",
    'tanh_sine_osc':    r"$x^2 + 5\tanh(100\sin(5x)) + 5\tanh(100\sin(7x))$",
}

display_names = {
    'sharp_tanh_steps': 'Sharp Tanh Steps',
    'fractal_sine':     'Fractal Sine',
    'tanh_sine_osc':    'Tanh-Sine Oscillations',
}

# Optimization routine

def optimize_1d(fn, optimizer_cls, init_val, lr, steps):
    theta = torch.nn.Parameter(torch.tensor(init_val, dtype=torch.float32))
    optimizer = optimizer_cls([theta], lr=lr)
    traj_theta, traj_loss = [], []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = fn(theta)
        loss.backward()
        optimizer.step()
        traj_theta.append(theta.item())
        traj_loss.append(loss.item())
    return np.array(traj_theta), np.array(traj_loss)

# Plotting with grid and LaTeX titles
def plot_comparison(fn_name, fn, results, x_range, save_dir):
    theta_diff, loss_diff = results['DiffGrad']
    theta_pulse, loss_pulse = results['PulseGrad']

    xs = np.linspace(*x_range, 500)
    ys = fn(xs)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Landscape with LaTeX equation
    axes[0].plot(xs, ys)
    axes[0].set_title(equations[fn_name], fontsize=14)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('f(x)')
    axes[0].grid(True)

    # Loss vs. Step
    axes[1].plot(loss_diff,  label='DiffGrad')
    axes[1].plot(loss_pulse, label='PulseGrad')
    axes[1].set_title('Loss vs. Step')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True)
    axes[1].legend()

    # Parameter vs. Step
    axes[2].plot(theta_diff,  label='DiffGrad')
    axes[2].plot(theta_pulse, label='PulseGrad')
    axes[2].set_title('Parameter (θ) vs. Step')
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel(r'$\theta$')
    axes[2].grid(True)
    axes[2].legend()

    # Figure title
    fig.suptitle(display_names[fn_name], fontsize=16)
    plt.tight_layout(rect=[0,0.03,1,0.95])

    # Save under ./results/
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f'{fn_name}_comparison.png')
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved comparison for {fn_name} at {out_path}")

# Main execution
def main():
    init_val = 2.5
    lr = 0.02
    steps = 150
    save_dir = './results'

    opt_fns = {
        'DiffGrad':  lambda p, lr: diffgrad(p, lr=lr),
        'PulseGrad': lambda p, lr: PulseGrad(p, lr=lr),
    }

    for name, fn in toy_functions.items():
        results = {}
        for opt_name, opt_cls in opt_fns.items():
            traj_t, traj_l = optimize_1d(fn, opt_cls, init_val, lr, steps)
            results[opt_name] = (traj_t, traj_l)
        plot_comparison(name, fn, results, (-5, 5), save_dir)

if __name__ == '__main__':
    main()
