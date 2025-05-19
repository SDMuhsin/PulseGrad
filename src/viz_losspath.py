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
def rapid_gradient_oscillation(x):
    """
    A function with a quadratic basin overlaid with a high-frequency cosine wave.
    The gradient includes a term like -A*B*sin(B*x), which oscillates rapidly
    in magnitude and sign, causing (g_t - g_{t-1})^2 to be frequently large.
    Experimental optimizer's enhanced v_t should help stabilize steps.

    f(x) = C*x^2 + A*cos(B*x)
    f'(x) = 2*C*x - A*B*sin(B*x)
    """
    A = 1.0   # Amplitude of cosine wave
    B = 50.0  # Frequency of cosine wave (high for rapid oscillation)
    C = 0.05  # Strength of the underlying quadratic basin
    def torch_fn(t):
        return C * t**2 + A * torch.cos(B * t)
    def np_fn(t):
        return C * t**2 + A * np.cos(B * t)
    return torch_fn(x) if isinstance(x, torch.Tensor) else np_fn(x)

def narrow_well_wide_plateau(x):
    """
    A function with a gentle global slope but a very narrow, deep Gaussian well.
    The gradient is small on the "plateau" but becomes very large and changes
    sign abruptly within the narrow well.
    If an optimizer approaches the well from the plateau (g_{t-1} is small)
    and encounters the steep slope of the well (g_t is large), the
    (g_{t-1} - g_t)^2 term in Experimental's v_t will be large. This inflates
    v_t (approx (1+gamma)*g_t^2 vs g_t^2 for Adam/diffGrad), leading to a
    more cautious step that might help Experimental settle into the well
    while others might overshoot.

    f(x) = D*(x+x_offset)^2 - A*exp(-k*(x-c)^2)
    f'(x) = 2*D*(x+x_offset) + 2*A*k*(x-c)*exp(-k*(x-c)^2)
    """
    A = 6.0       # Depth of the well
    k = 250.0     # Narrowness of the well (larger k = narrower)
    c = 2.0       # Center of the well
    D = 0.005     # Strength of the quadratic pull
    x_offset = 5.0 # Shifts the minimum of the quadratic term
    def torch_fn(t):
        return D * (t + x_offset)**2 - A * torch.exp(-k * (t - c)**2)
    def np_fn(t):
        return D * (t + x_offset)**2 - A * np.exp(-k * (t - c)**2)
    return torch_fn(x) if isinstance(x, torch.Tensor) else np_fn(x)

def erratic_sign_changes(x):
    """
    A function whose gradient is dominated by a high-frequency, high-amplitude
    sine wave, causing frequent and sharp sign changes in the gradient.
    The underlying structure is a gentle quadratic.
    When g_t approx -g_{t-1} (a sign flip), (g_t - g_{t-1})^2 approx (2*g_t)^2 = 4*g_t^2.
    Experimental's v_t update incorporates g_t^2 + gamma*(g_t - g_{t-1})^2,
    which becomes approx. (1+4*gamma)*g_t^2. This is significantly larger
    than Adam/diffGrad's v_t (based on g_t^2), leading to much smaller steps
    for Experimental, potentially improving stability and preventing oscillations.

    The function is derived from f(x) = D*x^2 + 0.5*A * [cos(k_diff*x) - cos(k_sum*x)]
    Its gradient is f'(x) = 2*D*x + 0.5*A * [-k_diff*sin(k_diff*x) + k_sum*sin(k_sum*x)]
    """
    A = 1.0      # Amplitude factor for oscillatory part
    k_sum = 22.0 # Corresponds to (k1+k2)
    k_diff = 2.0 # Corresponds to (k1-k2)
    D = 0.01     # Strength of the underlying quadratic basin
    def torch_fn(t):
        # This form's gradient: 2Dt + 0.5A(-k_diff sin(k_diff t) + k_sum sin(k_sum t))
        return D * t**2 + 0.5 * A * (torch.cos(k_diff * t) - torch.cos(k_sum * t))
    def np_fn(t):
        return D * t**2 + 0.5 * A * (np.cos(k_diff * t) - np.cos(k_sum * t))
    return torch_fn(x) if isinstance(x, torch.Tensor) else np_fn(x)

# Dictionary of the new toy functions
toy_functions = {
    "rapid_gradient_oscillation": rapid_gradient_oscillation,
    "narrow_well_wide_plateau":   narrow_well_wide_plateau,
    "erratic_sign_changes":       erratic_sign_changes,
}

equations = {
    "rapid_gradient_oscillation": r"$0.05 x^2 + \cos(50x)$",
    "narrow_well_wide_plateau":   r"$0.005(x+5)^2 - 6e^{-250(x-2)^2}$",
    "erratic_sign_changes":       r"$0.01 x^2 + 0.5(\cos(2x) - \cos(22x))$",
}

display_names = {
    "rapid_gradient_oscillation": "Rapid Gradient Oscillation",
    "narrow_well_wide_plateau":   "Narrow Well on Wide Plateau",
    "erratic_sign_changes":       "Erratic Sign Changes",
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

