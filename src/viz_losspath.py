#!/usr/bin/env python
"""
Example
-------
$ python optimizer_trajectory_1d_comparison.py \
      --optimizers adam,diffgrad,pulsegrad \
      --lr 0.02 --steps 150
"""
import math
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
    from experimental.exp2 import ExperimentalV2 as PulseGradV2
except ImportError:                                        # noqa: D401,E501
    PulseGrad = None
# ──────────────────────────────────────────────────────────────────
# 1-D TOY FUNCTIONS
# ------------------------------------------------------------------
def high_frequency_gauntlet(x, A_env_amp=1.0, k_env_width=5.0, x_mod_center=1.0, B_osc_freq=100.0, D_quad_pull=0.001):
    """
    An intense, localized region of very high-frequency oscillations with
    Gaussian-modulated amplitude, overlaying a gentle global pull.
    f(x) = A_env_amp * exp(-k_env_width*(x-x_mod_center)^2) * cos(B_osc_freq*x) + D_quad_pull*x^2
    """
    envelope = A_env_amp * torch.exp(-k_env_width * (x - x_mod_center)**2) if isinstance(x, torch.Tensor) else A_env_amp * np.exp(-k_env_width * (x - x_mod_center)**2)
    
    if isinstance(x, torch.Tensor):
        oscillation_term = envelope * torch.cos(B_osc_freq * x)
        quadratic_term = D_quad_pull * x**2
        return oscillation_term + quadratic_term
    else: # numpy
        oscillation_term = envelope * np.cos(B_osc_freq * x)
        quadratic_term = D_quad_pull * x**2
        return oscillation_term + quadratic_term

def mesa_trap(x, D_pull=0.01, x_true_target=-2.0, A_mesa=0.5, k_mesa_width=8.0, x_mesa_center=0.75, N_mesa_power=4):
    """
    A global quadratic pull. A very flat-topped "mesa" (plateau) is on the path.
    Gradient onto mesa is steep. Gradient on top is tiny. Adam's v_t may decay,
    leading to a large step off. Experimental's v_t should stay larger due to diff.
    f(x) = D_pull*(x-x_true_target)^2 + A_mesa * exp(-(k_mesa_width*(x-x_mesa_center))**(2*N_mesa_power))
    """
    u_mesa = k_mesa_width * (x - x_mesa_center)
    if isinstance(x, torch.Tensor):
        quadratic_term = D_pull * (x - x_true_target)**2
        # Power needs to be applied carefully for negative bases if N_mesa_power is not integer
        # However, (u_mesa)^even_power is same as (|u_mesa|)^even_power
        # Or ensure u_mesa is passed to pow after abs if that was the intent for robustness,
        # but mathematically u_mesa**(2*N) implies the sign of u_mesa is lost.
        # For u_mesa^8, sign is lost.
        mesa_term = A_mesa * torch.exp(-torch.pow(u_mesa, 2 * N_mesa_power))
        return quadratic_term + mesa_term
    else: # numpy
        quadratic_term = D_pull * (x - x_true_target)**2
        # mesa_term = A_mesa * np.exp(-(np.abs(u_mesa)**(2 * N_mesa_power))) # If N_mesa_power could be non-integer
        mesa_term = A_mesa * np.exp(-(u_mesa**(2 * N_mesa_power)))
        return quadratic_term + mesa_term

def decoy_basin_turbulent_exit(x, D_global=0.01, x_target=-2.0,
                               A_decoy=0.3, k_decoy=10.0, x_decoy_center=0.5,
                               A_subtle=0.03, F_subtle=15.0,
                               A_ramp=0.75, F_ramp=120.0, x_ramp_center=-0.25, k_ramp_env=80.0):
    """
    Lures optimizer into a shallow decoy basin with subtle oscillations, then requires
    navigating a short, highly turbulent 'exit ramp' to reach the true minimum.
    Adam may take a large step into turbulence; Experimental should adapt better.
    """
    if isinstance(x, torch.Tensor):
        # Global pull
        term_global_pull = D_global * (x - x_target)**2
        
        # Decoy Basin
        u_decoy = x - x_decoy_center
        term_decoy_well = -A_decoy * torch.exp(-k_decoy * u_decoy**2)
        term_subtle_osc = A_subtle * torch.cos(F_subtle * u_decoy) # Centered with decoy
        
        # Turbulent Exit Ramp
        u_ramp = x - x_ramp_center
        term_ramp_osc = A_ramp * torch.cos(F_ramp * u_ramp)
        term_ramp_envelope = torch.exp(-k_ramp_env * u_ramp**2)
        term_turbulent_ramp = term_ramp_osc * term_ramp_envelope
        
        return term_global_pull + term_decoy_well + term_subtle_osc + term_turbulent_ramp
    else: # numpy
        term_global_pull = D_global * (x - x_target)**2
        
        u_decoy = x - x_decoy_center
        term_decoy_well = -A_decoy * np.exp(-k_decoy * u_decoy**2)
        term_subtle_osc = A_subtle * np.cos(F_subtle * u_decoy)
        
        u_ramp = x - x_ramp_center
        term_ramp_osc = A_ramp * np.cos(F_ramp * u_ramp)
        term_ramp_envelope = np.exp(-k_ramp_env * u_ramp**2)
        term_turbulent_ramp = term_ramp_osc * term_ramp_envelope
        
        return term_global_pull + term_decoy_well + term_subtle_osc + term_turbulent_ramp

# Dictionary of the toy functions
toy_functions = {
    "high_frequency_gauntlet": high_frequency_gauntlet,
    "mesa_trap": mesa_trap,
    "decoy_basin_turbulent_exit": decoy_basin_turbulent_exit,
}

# LaTeX representations of the functions
equations = {
    "high_frequency_gauntlet": r"$A_e e^{-k_e(x-x_c)^2} \cos(Bx) + D x^2$",
    "mesa_trap": r"$D(x-x_t)^2 + A_m e^{-(k_m(x-x_m))^{2N}}$",
    "decoy_basin_turbulent_exit": r"$D(x-x_t)^2 - A_d e^{-k_d(x-x_d)^2} + A_s \cos(F_s(x-x_d)) + A_r \cos(F_r(x-x_r))e^{-k_r(x-x_r)^2}$",
}

# Display names for plots or tables
display_names = {
    "high_frequency_gauntlet": "High-Frequency Gauntlet",
    "mesa_trap": "Mesa Trap",
    "decoy_basin_turbulent_exit": "Decoy Basin & Turbulent Exit",
}
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

if PulseGradV2 is not None:
    AVAILABLE_OPTIMIZERS["pulsegradv2"] = PulseGradV2
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

