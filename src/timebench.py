#!/usr/bin/env python
# coding=utf-8

import argparse
import time
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from pathlib import Path
import csv
import statistics # Keep for potential future use, though direct median is fine for now

# Optimizer imports
from experimental.eexp import ExperimentalOptimized as PulseGrad # Renamed as requested
# Assuming these are in the PYTHONPATH or current directory structure as in the reference
# If 'experimental' is a directory, ensure __init__.py is present
# For stubs, if they are truly just optimizers, they should accept params and lr.
# If they have complex dependencies, those need to be available.

try:
    from experimental.diffgrad import diffgrad
except ImportError:
    print("Warning: diffgrad not found. It will be skipped in benchmarks.")
    diffgrad = None # Placeholder
try:
    from adabelief_pytorch import AdaBelief
except ImportError:
    print("Warning: AdaBelief not found. It will be skipped in benchmarks.")
    AdaBelief = None
try:
    from adamp import AdamP
except ImportError:
    print("Warning: AdamP not found. It will be skipped in benchmarks.")
    AdamP = None
try:
    import madgrad
except ImportError:
    print("Warning: madgrad not found. It will be skipped in benchmarks.")
    madgrad = None
try:
    from adan_pytorch import Adan
except ImportError:
    print("Warning: Adan not found. It will be skipped in benchmarks.")
    Adan = None
try:
    from lion_pytorch import Lion
except ImportError:
    print("Warning: Lion not found. It will be skipped in benchmarks.")
    Lion = None


# --- Toy Model Definition ---
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*layers)
        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        return self.network(x)

# --- Optimizer Instantiation ---
def get_optimizer_instance(optimizer_name, model_params, lr=1e-3):
    """
    Return a PyTorch optimizer based on its name.
    """
    optimizer_name = optimizer_name.lower()
    if optimizer_name == 'adagrad':
        return optim.Adagrad(model_params, lr=lr)
    elif optimizer_name == 'adadelta':
        return optim.Adadelta(model_params, lr=lr)
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(model_params, lr=lr)
    elif optimizer_name == 'amsgrad':
        return optim.Adam(model_params, lr=lr, amsgrad=True)
    elif optimizer_name == 'adam':
        return optim.Adam(model_params, lr=lr)
    elif optimizer_name == 'sgd':
        return optim.SGD(model_params, lr=lr)
    elif optimizer_name == 'pulsegrad':
        return PulseGrad(model_params, lr=lr)
    elif optimizer_name == 'diffgrad' and diffgrad:
        return diffgrad(model_params, lr=lr)
    elif optimizer_name == 'adabelief' and AdaBelief:
        return AdaBelief(model_params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decouple=True, rectify=True)
    elif optimizer_name == 'adamp' and AdamP:
        return AdamP(model_params, lr=lr, betas=(0.9, 0.999), weight_decay=1e-2)
    elif optimizer_name == 'madgrad' and madgrad:
        return madgrad.MADGRAD(model_params, lr=lr, momentum=0.9, weight_decay=0) # Simplified, adjust if needed
    elif optimizer_name == 'adan' and Adan:
        return Adan(model_params, lr=lr, betas=(0.98, 0.92, 0.99), weight_decay=0.0) # Simplified
    elif optimizer_name == 'lion' and Lion:
        return Lion(model_params, lr=lr, weight_decay=0.0) # Simplified
    else:
        if optimizer_name in ['diffgrad', 'adabelief', 'adamp', 'madgrad', 'adan', 'lion']:
            print(f"Optimizer {optimizer_name} was requested but not found/imported. Skipping.")
            return None
        raise ValueError(f"Unknown or unavailable optimizer: {optimizer_name}")

# --- Benchmarking Core ---
def benchmark_optimizer_single_run(model, optimizer_name, device, num_steps=100, batch_size=64, input_features=128, num_classes=10, lr=1e-3):
    """
    Benchmarks a single optimizer on a given model and device.
    """
    model.to(device)
    optimizer = get_optimizer_instance(optimizer_name, model.parameters(), lr=lr)
    if optimizer is None: # Optimizer unavailable
        return None

    criterion = nn.CrossEntropyLoss()

    # Dummy data
    inputs = torch.randn(batch_size, input_features).to(device)
    labels = torch.randint(0, num_classes, (batch_size,)).to(device)

    # CPU Memory Tracking
    process = psutil.Process(os.getpid())
    cpu_mem_before = process.memory_info().rss / (1024 * 1024) # MB

    # GPU Memory Tracking (if applicable)
    gpu_mem_allocated_before = 0
    gpu_max_mem_allocated_before = 0
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        gpu_mem_allocated_before = torch.cuda.memory_allocated(device) / (1024 * 1024) # MB
        gpu_max_mem_allocated_before = torch.cuda.max_memory_allocated(device) / (1024 * 1024) # MB

    # Warm-up (optional, but good for stable time measurements)
    for _ in range(min(10, num_steps // 10)): # Warm up for a few steps
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Actual Benchmark
    start_time = time.time()
    for _ in range(num_steps):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    end_time = time.time()

    wall_clock_time = (end_time - start_time) / num_steps # Average time per step

    cpu_mem_after = process.memory_info().rss / (1024 * 1024) # MB
    cpu_mem_used = cpu_mem_after - cpu_mem_before # This is a rough estimate of increase

    gpu_mem_allocated_after = 0
    gpu_max_mem_allocated_after = 0
    if device.type == 'cuda':
        gpu_mem_allocated_after = torch.cuda.memory_allocated(device) / (1024 * 1024) # MB
        gpu_max_mem_allocated_after = torch.cuda.max_memory_allocated(device) / (1024 * 1024) # MB

    return {
        "optimizer": optimizer_name,
        "device": device.type,
        "model_params": model.num_params,
        "avg_step_time_ms": wall_clock_time * 1000,
        "peak_cpu_mem_usage_mb": cpu_mem_after, # Total process RSS
        "peak_gpu_mem_allocated_mb": gpu_max_mem_allocated_after if device.type == 'cuda' else 0,
    }

# --- Experiment Orchestration ---
def run_experiments(optimizer_names, model_configs, devices, num_steps, batch_size, input_features, num_classes, lr, results_dir):
    all_results = []
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    for model_name, config in model_configs.items():
        print(f"\n--- Benchmarking Model: {model_name} (Params: {SimpleMLP(input_features, config['hidden_size'], config['num_layers'], num_classes).num_params}) ---")
        for device_str in devices:
            device = torch.device(device_str)
            if device.type == 'cuda' and not torch.cuda.is_available():
                print(f"CUDA requested but not available. Skipping {device_str} for model {model_name}.")
                continue
            print(f"  Device: {device.type.upper()}")

            model_instance = SimpleMLP(input_features, config['hidden_size'], config['num_layers'], num_classes)

            for opt_name in optimizer_names:
                # Check if optimizer is available
                temp_opt = get_optimizer_instance(opt_name, model_instance.parameters(), lr)
                if temp_opt is None and opt_name not in ['adagrad', 'adadelta', 'rmsprop', 'amsgrad', 'adam', 'sgd', 'pulsegrad']: # built-in or pulsegrad
                    print(f"    Optimizer {opt_name} is not available. Skipping.")
                    continue
                del temp_opt # clean up

                print(f"    Benchmarking Optimizer: {opt_name}...")
                try:
                    # Multiple runs for stability
                    num_runs = 3
                    run_results_list = []
                    for i in range(num_runs):
                        print(f"      Run {i+1}/{num_runs}...")
                        # Re-create model for fresh optimizer state and memory stats if needed,
                        # though for this benchmark, the model itself doesn't change much.
                        # Crucially, GPU memory stats should be reset if possible.
                        current_model = SimpleMLP(input_features, config['hidden_size'], config['num_layers'], num_classes)
                        res = benchmark_optimizer_single_run(
                            current_model, opt_name, device,
                            num_steps, batch_size, input_features, num_classes, lr
                        )
                        if res:
                            run_results_list.append(res)
                        if device.type == 'cuda': # Ensure clean slate for next GPU run
                            del current_model
                            torch.cuda.empty_cache()


                    if not run_results_list:
                        continue

                    # Aggregate results (e.g., median for time, max for memory)
                    # For simplicity, we'll take the median of times and max of memory.
                    # More sophisticated: median for all, or mean +/- std.
                    # Let's use median for times and max for memory, which seems reasonable.
                    final_res = run_results_list[0].copy() # template
                    final_res["avg_step_time_ms"] = statistics.median([r["avg_step_time_ms"] for r in run_results_list])
                    final_res["peak_cpu_mem_usage_mb"] = max([r["peak_cpu_mem_usage_mb"] for r in run_results_list])
                    if device.type == 'cuda':
                         final_res["peak_gpu_mem_allocated_mb"] = max([r["peak_gpu_mem_allocated_mb"] for r in run_results_list])
                    final_res["model_config_name"] = model_name # For easier grouping

                    all_results.append(final_res)
                    print(f"      Done. Avg Step Time: {final_res['avg_step_time_ms']:.2f}ms, Peak GPU Mem: {final_res['peak_gpu_mem_allocated_mb']:.2f}MB")

                except Exception as e:
                    print(f"    ERROR benchmarking {opt_name} on {model_name} with {device.type}: {e}")
                    import traceback
                    traceback.print_exc()

    return pd.DataFrame(all_results)

# --- Output Generation ---
def generate_tables_and_plots(df_results, results_dir):
    if df_results.empty:
        print("No results to generate tables/plots.")
        return

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # --- Wall Clock Time ---
    # Table
    time_table_data = []
    for device_type in df_results['device'].unique():
        df_device = df_results[df_results['device'] == device_type]
        pivot_time = df_device.pivot_table(index="optimizer", columns="model_config_name", values="avg_step_time_ms")
        pivot_time = pivot_time.round(2)
        pivot_time.columns = [f"{col} (ms)" for col in pivot_time.columns]
        pivot_time.to_csv(Path(results_dir) / f"wall_clock_time_{device_type}.csv")
        print(f"\nWall Clock Time (ms) - {device_type.upper()}:\n", pivot_time)
        time_table_data.append(pivot_time.reset_index().rename(columns={'optimizer': f'Optimizer ({device_type.upper()})'}))

    # Plot (Scalability of Time)
    plt.figure(figsize=(12, 7))
    plot_df_time = df_results.copy()
    plot_df_time['Model Parameters (log scale)'] = np.log10(plot_df_time['model_params']) # Use log for x-axis if param range is large
    g = sns.lineplot(data=plot_df_time, x="model_params", y="avg_step_time_ms", hue="optimizer", style="device", marker="o", markersize=8, linewidth=2)
    plt.xlabel("Number of Model Parameters", fontsize=14)
    plt.ylabel("Average Step Time (ms)", fontsize=14)
    plt.title("Optimizer Wall Clock Time vs. Model Size", fontsize=16, fontweight='bold')
    plt.legend(title="Optimizer | Device", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.xscale('log') # Better for wide range of params
    plt.yscale('log') # Often step time also scales non-linearly
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(Path(results_dir) / "scalability_wall_clock_time.png", dpi=300)
    plt.savefig(Path(results_dir) / "scalability_wall_clock_time.pdf", dpi=300)
    print(f"\nSaved plot: {Path(results_dir) / 'scalability_wall_clock_time.png'}")
    plt.close()


    # --- Memory Usage ---
    # GPU Memory Table & Plot (if GPU results exist)
    df_gpu = df_results[df_results['device'] == 'cuda']
    if not df_gpu.empty:
        pivot_gpu_mem = df_gpu.pivot_table(index="optimizer", columns="model_config_name", values="peak_gpu_mem_allocated_mb")
        pivot_gpu_mem = pivot_gpu_mem.round(2)
        pivot_gpu_mem.columns = [f"{col} (MB)" for col in pivot_gpu_mem.columns]
        pivot_gpu_mem.to_csv(Path(results_dir) / "peak_gpu_memory.csv")
        print(f"\nPeak GPU Memory (MB):\n", pivot_gpu_mem)

        plt.figure(figsize=(12, 7))
        plot_df_gpu_mem = df_gpu.copy()
        g = sns.lineplot(data=plot_df_gpu_mem, x="model_params", y="peak_gpu_mem_allocated_mb", hue="optimizer", marker="o", markersize=8, linewidth=2)
        plt.xlabel("Number of Model Parameters", fontsize=14)
        plt.ylabel("Peak GPU Memory Allocated (MB)", fontsize=14)
        plt.title("Optimizer GPU Memory vs. Model Size", fontsize=16, fontweight='bold')
        plt.legend(title="Optimizer", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.xscale('log')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(Path(results_dir) / "scalability_gpu_memory.png", dpi=300)
        plt.savefig(Path(results_dir) / "scalability_gpu_memory.pdf", dpi=300)
        print(f"\nSaved plot: {Path(results_dir) / 'scalability_gpu_memory.png'}")
        plt.close()

    # CPU Memory Table & Plot
    # Note: CPU memory is trickier as it's total process RSS. Differences might be subtle or noisy.
    df_cpu_mem = df_results[df_results['device'] == 'cpu'] # Or analyze for both, if CPU part of optimizer state is significant
    if not df_cpu_mem.empty : # Focus on CPU device runs, or all if you want to see overhead on GPU runs too
        pivot_cpu_mem = df_cpu_mem.pivot_table(index="optimizer", columns="model_config_name", values="peak_cpu_mem_usage_mb")
        pivot_cpu_mem = pivot_cpu_mem.round(2)
        pivot_cpu_mem.columns = [f"{col} (MB)" for col in pivot_cpu_mem.columns]
        pivot_cpu_mem.to_csv(Path(results_dir) / "peak_cpu_memory_process.csv")
        print(f"\nPeak CPU Memory (Process RSS, MB) - for CPU runs:\n", pivot_cpu_mem)

        plt.figure(figsize=(12, 7))
        plot_df_cpu_mem = df_cpu_mem.copy()
        g = sns.lineplot(data=plot_df_cpu_mem, x="model_params", y="peak_cpu_mem_usage_mb", hue="optimizer", marker="o", markersize=8, linewidth=2)
        plt.xlabel("Number of Model Parameters", fontsize=14)
        plt.ylabel("Peak Process CPU Memory (RSS, MB)", fontsize=14)
        plt.title("Optimizer CPU Memory (Process) vs. Model Size (CPU Runs)", fontsize=16, fontweight='bold')
        plt.legend(title="Optimizer", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.xscale('log')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(Path(results_dir) / "scalability_cpu_memory.png", dpi=300)
        plt.savefig(Path(results_dir) / "scalability_cpu_memory.pdf", dpi=300)
        print(f"\nSaved plot: {Path(results_dir) / 'scalability_cpu_memory.png'}")
        plt.close()

    print(f"\nAll tables and plots saved to: {Path(results_dir).resolve()}")


# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch Optimizers")
    parser.add_argument(
        "--optimizers",
        type=str,
        nargs="+",
        default=["sgd", "adam", "pulsegrad", "adabelief", "lion", "adan"], # Add more as needed and available
        help="List of optimizer names to benchmark."
    )
    parser.add_argument(
        "--devices",
        type=str,
        nargs="+",
        default=["cpu", "cuda"],
        help="List of devices to benchmark on ('cpu', 'cuda')."
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=200, # Increased for better averaging after warmup
        help="Number of optimizer steps per run."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for dummy data."
    )
    parser.add_argument(
        "--input_features",
        type=int,
        default=256,
        help="Input features for the MLP model."
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=10,
        help="Output classes for the MLP model."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for optimizers."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results_optimizer_benchmark",
        help="Directory to save tables and plots."
    )
    parser.add_argument(
        "--model_hidden_sizes",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512, 1024], # Hidden sizes for different model scales
        help="List of hidden sizes for MLP layers for scalability testing."
    )
    parser.add_argument(
        "--model_num_layers",
        type=int,
        nargs="+",
        default=[2, 4, 8], # Number of layers for different model scales
        help="List of number of hidden layers for MLP for scalability testing."
    )

    args = parser.parse_args()

    # Define model configurations for scalability
    # Creates combinations of hidden sizes and num_layers
    model_configs = {}
    id_counter = 0
    for hs in args.model_hidden_sizes:
        for nl in args.model_num_layers:
            # Calculate approx params for naming, actual params calculated in SimpleMLP
            # This is just for a somewhat informative name.
            # Actual param count will be in the results.
            model_configs[f"MLP_h{hs}_l{nl}"] = {"hidden_size": hs, "num_layers": nl}


    # Filter out optimizers that couldn't be imported
    available_optimizers = []
    for opt_name in args.optimizers:
        opt_name_lower = opt_name.lower()
        if opt_name_lower == 'pulsegrad' or opt_name_lower in ['sgd', 'adam', 'adagrad', 'adadelta', 'rmsprop', 'amsgrad']:
            available_optimizers.append(opt_name)
        elif opt_name_lower == 'diffgrad' and diffgrad:
            available_optimizers.append(opt_name)
        elif opt_name_lower == 'adabelief' and AdaBelief:
            available_optimizers.append(opt_name)
        elif opt_name_lower == 'adamp' and AdamP:
            available_optimizers.append(opt_name)
        elif opt_name_lower == 'madgrad' and madgrad:
            available_optimizers.append(opt_name)
        elif opt_name_lower == 'adan' and Adan:
            available_optimizers.append(opt_name)
        elif opt_name_lower == 'lion' and Lion:
            available_optimizers.append(opt_name)
        else:
            print(f"Optimizer '{opt_name}' is not available or was not imported correctly, skipping.")
    
    if not available_optimizers:
        print("No optimizers available to benchmark. Exiting.")
        return

    print("Starting optimizer benchmarks with the following configurations:")
    print(f"  Optimizers: {available_optimizers}")
    print(f"  Devices: {args.devices}")
    print(f"  Model Configurations (HiddenSize_NumLayers): {list(model_configs.keys())}")
    print(f"  Num Steps: {args.num_steps}, Batch Size: {args.batch_size}")

    results_df = run_experiments(
        optimizer_names=available_optimizers,
        model_configs=model_configs,
        devices=args.devices,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        input_features=args.input_features,
        num_classes=args.num_classes,
        lr=args.lr,
        results_dir=args.results_dir
    )

    if results_df is not None and not results_df.empty:
        generate_tables_and_plots(results_df, args.results_dir)
    else:
        print("No results were generated from the benchmark runs.")

    print("\nBenchmarking script finished.")

if __name__ == "__main__":
    main()
