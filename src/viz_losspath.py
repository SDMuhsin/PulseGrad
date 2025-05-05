#!/usr/bin/env python
"""
Example
-------
$ python optimizer_trajectory_1d_comparison.py \
      --optimizers adam,diffgrad,pulsegrad \
      --lr 0.02 --steps 150
"""
import argparse
import os
import warnings
from typing import Dict, Callable, Sequence

import numpy as np
import torch
import matplotlib.pyplot as plt

# Silence the harmless torch warning triggered inside some custom opts
warnings.filterwarnings("ignore", message="This overload of add_ is deprecated")

# ──────────────────────────────────────────────────────────────────
# OPTIONAL / THIRD-PARTY OPTIMIZERS
# ------------------------------------------------------------------
# DiffGrad & PulseGrad live in your repo’s experimental/ directory.
# Import errors are swallowed so the script still works when they
# aren’t available.
try:
    from experimental.diffgrad import diffgrad            # noqa: E402
except ImportError:                                        # noqa: D401,E501
    diffgrad = None

try:
    from experimental.exp import Experimental as PulseGrad  # noqa: E402
except ImportError:                                        # noqa: D401,E501
    PulseGrad = None
# ──────────────────────────────────────────────────────────────────
# 1-D TOY FUNCTIONS
# ------------------------------------------------------------------
def sharp_tanh_steps(x):
    def torch_fn(t):
        return 0.5 * t**2 + 5 * torch.tanh(100 * (t - 1)) - 5 * torch.tanh(100 * (t - 2))
    def np_fn(t):
        return 0.5 * t**2 + 5 * np.tanh(100 * (t - 1)) - 5 * np.tanh(100 * (t - 2))
    return torch_fn(x) if isinstance(x, torch.Tensor) else np_fn(x)

def fractal_sine(x):
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
    def torch_fn(t):
        return t**2 + 5 * torch.tanh(100 * torch.sin(5 * t)) + 5 * torch.tanh(100 * torch.sin(7 * t))
    def np_fn(t):
        return t**2 + 5 * np.tanh(100 * np.sin(5 * t)) + 5 * np.tanh(100 * np.sin(7 * t))
    return torch_fn(x) if isinstance(x, torch.Tensor) else np_fn(x)

toy_functions = {
    "sharp_tanh_steps": sharp_tanh_steps,
    "fractal_sine":     fractal_sine,
    "tanh_sine_osc":    tanh_sine_osc,
}

equations = {
    "sharp_tanh_steps": r"$0.5 x^2 + 5\tanh(100(x-1)) - 5\tanh(100(x-2))$",
    "fractal_sine":     r"$\sum_{i=1}^{10}2^{-i}\sin(2^i x) + 0.02 x^2$",
    "tanh_sine_osc":    r"$x^2 + 5\tanh(100\sin(5x)) + 5\tanh(100\sin(7x))$",
}

display_names = {
    "sharp_tanh_steps": "Sharp Tanh Steps",
    "fractal_sine":     "Fractal Sine",
    "tanh_sine_osc":    "Tanh-Sine Oscillations",
}
# ──────────────────────────────────────────────────────────────────
# OPTIMIZATION CORE
# ------------------------------------------------------------------
def optimize_1d(
    fn: Callable[[torch.Tensor], torch.Tensor],
    optimizer_cls: Callable,
    init_val: float,
    lr: float,
    steps: int
) -> tuple[np.ndarray, np.ndarray]:
    θ = torch.nn.Parameter(torch.tensor(init_val, dtype=torch.float32))
    optimizer = optimizer_cls([θ], lr=lr)
    θ_traj, loss_traj = [], []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = fn(θ)
        loss.backward()
        optimizer.step()
        θ_traj.append(float(θ))
        loss_traj.append(float(loss))
    return np.asarray(θ_traj), np.asarray(loss_traj)
# ──────────────────────────────────────────────────────────────────
# PLOTTING
# ------------------------------------------------------------------
def plot_comparison(
    fn_key: str,
    fn: Callable,
    results: Dict[str, tuple[np.ndarray, np.ndarray]],
    x_range: Sequence[float],
    save_dir: str
) -> None:
    xs  = np.linspace(*x_range, 500)
    ys  = fn(xs)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Function landscape
    axes[0].plot(xs, ys)
    axes[0].set_title(equations[fn_key], fontsize=14)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("f(x)")
    axes[0].grid(True)

    # Trajectories
    for name, (θ_traj, loss_traj) in results.items():
        axes[1].plot(loss_traj, label=name)
        axes[2].plot(θ_traj,    label=name)

    axes[1].set_title("Loss vs. Step")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].set_title("Parameter (θ) vs. Step")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel(r"$\theta$")
    axes[2].grid(True)
    axes[2].legend()

    fig.suptitle(display_names[fn_key], fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    os.makedirs(save_dir, exist_ok=True)
    fname = f"{fn_key}_{'_'.join(results)}.png"
    plt.savefig(os.path.join(save_dir, fname))
    plt.close(fig)
    print(f"Saved → {save_dir}/{fname}")
# ──────────────────────────────────────────────────────────────────
# AVAILABLE OPTIMIZERS MAP
# ------------------------------------------------------------------
AVAILABLE_OPTIMIZERS: Dict[str, Callable] = {
    "sgd":      torch.optim.SGD,
    "adam":     torch.optim.Adam,
    "adamw":    torch.optim.AdamW,
    "rmsprop":  torch.optim.RMSprop,
}
if diffgrad is not None:
    AVAILABLE_OPTIMIZERS["diffgrad"] = diffgrad
if PulseGrad is not None:
    AVAILABLE_OPTIMIZERS["pulsegrad"] = PulseGrad
# ──────────────────────────────────────────────────────────────────
# CLI & MAIN
# ------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare 1-D optimizer trajectories on toy functions")
    p.add_argument("--optimizers", type=str,
                   default="adam,diffgrad,pulsegrad",
                   help="Comma-separated list of optimizers to try")
    p.add_argument("--init",   type=float, default=2.5, help="Initial θ value")
    p.add_argument("--lr",     type=float, default=0.02, help="Learning rate")
    p.add_argument("--steps",  type=int,   default=150,  help="Number of steps")
    p.add_argument("--outdir", type=str,   default="./results",
                   help="Directory to store plots")
    return p.parse_args()

def main() -> None:
    args = parse_args()

    requested = [s.strip().lower() for s in args.optimizers.split(",") if s.strip()]
    unknown   = sorted(set(requested) - AVAILABLE_OPTIMIZERS.keys())
    if unknown:
        raise ValueError(f"Unknown optimizer(s): {', '.join(unknown)}")

    # Build the subset dict the rest of the script will use
    optimizers = {name: AVAILABLE_OPTIMIZERS[name] for name in requested}

    for fn_key, fn in toy_functions.items():
        results = {}
        for name, opt_cls in optimizers.items():
            θ_traj, loss_traj = optimize_1d(fn, opt_cls,
                                            args.init, args.lr, args.steps)
            results[name] = (θ_traj, loss_traj)
        plot_comparison(fn_key, fn, results, (-5, 5), args.outdir)

if __name__ == "__main__":
    main()

