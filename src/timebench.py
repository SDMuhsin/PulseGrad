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
import statistics
from adamp import AdamP
import madgrad
from adan_pytorch import Adan
from lion_pytorch import Lion

# Local experimental optimizers
from experimental.lance import LANCE
from experimental.diffgrad import diffgrad
from experimental.cosmic import COSMIC
from experimental.exp import Experimental
from experimental.exp2 import ExperimentalV2
from experimental.fused_exp import Experimental as PulseGrad

# Optimizer imports with error handling
try:
    from experimental.diffgrad import diffgrad
except ImportError:
    print("Warning: diffgrad not found. It will be skipped in benchmarks.")
    diffgrad = None
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
    """Return a PyTorch optimizer based on its name."""
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
        return madgrad.MADGRAD(model_params, lr=lr, momentum=0.9, weight_decay=0)
    elif optimizer_name == 'adan' and Adan:
        return Adan(model_params, lr=lr, betas=(0.98, 0.92, 0.99), weight_decay=0.0)
    elif optimizer_name == 'lion' and Lion:
        return Lion(model_params, lr=lr, weight_decay=0.0)
    else:
        if optimizer_name in ['diffgrad', 'adabelief', 'adamp', 'madgrad', 'adan', 'lion']:
            print(f"Optimizer {optimizer_name} was requested but not found/imported. Skipping.")
            return None
        raise ValueError(f"Unknown or unavailable optimizer: {optimizer_name}")

# --- Benchmarking Core ---
def benchmark_optimizer_single_run(model, optimizer_name, device, num_steps=100, batch_size=64, input_features=128, num_classes=10, lr=1e-3):
    """Benchmarks a single optimizer on a given model and device."""
    model.to(device)
    optimizer = get_optimizer_instance(optimizer_name, model.parameters(), lr=lr)
    if optimizer is None:
        return None

    criterion = nn.CrossEntropyLoss()
    inputs = torch.randn(batch_size, input_features).to(device)
    labels = torch.randint(0, num_classes, (batch_size,)).to(device)
    process = psutil.Process(os.getpid())
    cpu_mem_before = process.memory_info().rss / (1024 * 1024)

    gpu_mem_allocated_before = 0
    gpu_max_mem_allocated_before = 0
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        gpu_mem_allocated_before = torch.cuda.memory_allocated(device) / (1024 * 1024)
        gpu_max_mem_allocated_before = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    for _ in range(min(10, num_steps // 10)):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    start_time = time.time()
    for _ in range(num_steps):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    end_time = time.time()

    wall_clock_time = (end_time - start_time) / num_steps
    cpu_mem_after = process.memory_info().rss / (1024 * 1024)
    cpu_mem_used = cpu_mem_after - cpu_mem_before

    gpu_mem_allocated_after = 0
    gpu_max_mem_allocated_after = 0
    if device.type == 'cuda':
        gpu_mem_allocated_after = torch.cuda.memory_allocated(device) / (1024 * 1024)
        gpu_max_mem_allocated_after = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    return {
        "optimizer": optimizer_name.capitalize(),
        "device": device.type,
        "model_params": model.num_params,
        "avg_step_time_ms": wall_clock_time * 1000,
        "peak_cpu_mem_usage_mb": cpu_mem_after,
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
                temp_opt = get_optimizer_instance(opt_name, model_instance.parameters(), lr)
                if temp_opt is None and opt_name not in ['adagrad', 'adadelta', 'rmsprop', 'amsgrad', 'adam', 'sgd', 'pulsegrad']:
                    print(f"    Optimizer {opt_name} is not available. Skipping.")
                    continue
                del temp_opt

                print(f"    Benchmarking Optimizer: {opt_name}...")
                try:
                    num_runs = 1000
                    run_results_list = []
                    for i in range(num_runs):
                        print(f"          Run {i+1}/{num_runs}...")
                        current_model = SimpleMLP(input_features, config['hidden_size'], config['num_layers'], num_classes)
                        res = benchmark_optimizer_single_run(
                            current_model, opt_name, device,
                            num_steps, batch_size, input_features, num_classes, lr
                        )
                        if res:
                            run_results_list.append(res)
                        if device.type == 'cuda':
                            del current_model
                            torch.cuda.empty_cache()

                    if not run_results_list:
                        continue

                    for res in run_results_list:
                        res["model_config_name"] = model_name
                    all_results.extend(run_results_list)

                    median_time = statistics.median([r["avg_step_time_ms"] for r in run_results_list])
                    peak_gpu_mem = 0
                    if device.type == 'cuda' and any("peak_gpu_mem_allocated_mb" in r for r in run_results_list):
                        peak_gpu_mem = max([r["peak_gpu_mem_allocated_mb"] for r in run_results_list])

                    print(f"            Done. Median Step Time: {median_time:.2f}ms, Peak GPU Mem: {peak_gpu_mem:.2f}MB")

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

    # Define font sizes for camera-ready plots
    title_fontsize = 20
    label_fontsize = 18
    legend_fontsize = 14
    tick_fontsize = 14

    # Wall Clock Time Table
    summary_df_time = df_results.groupby(['device', 'optimizer', 'model_config_name', 'model_params'])['avg_step_time_ms'].median().reset_index()
    for device_type in summary_df_time['device'].unique():
        df_device_summary = summary_df_time[summary_df_time['device'] == device_type]
        pivot_time = df_device_summary.pivot_table(index="optimizer", columns="model_config_name", values="avg_step_time_ms")
        pivot_time = pivot_time.round(2)
        pivot_time.columns = [f"{col} (ms)" for col in pivot_time.columns]
        pivot_time.to_csv(Path(results_dir) / f"wall_clock_time_{device_type}.csv")
        print(f"\nWall Clock Time (Median, ms) - {device_type.upper()}:\n", pivot_time)

    # Wall Clock Time Plot
    for device_type in df_results['device'].unique():
        plt.figure(figsize=(12, 8)) # Increased figure size for better readability
        df_device = df_results[df_results['device'] == device_type].copy()
        
        sns.lineplot(
            data=df_device,
            x="model_params",
            y="avg_step_time_ms",
            hue="optimizer",
            marker="o",
            markersize=8,
            linewidth=2,
            errorbar="sd"  # Explicitly use standard deviation for error bands
        )

        plt.xlabel("Number of Model Parameters", fontsize=label_fontsize)
        plt.ylabel("Average Step Time (ms)", fontsize=label_fontsize)
        plt.title(f"Optimizer Wall Clock Time vs. Model Size ({device_type.upper()})", fontsize=title_fontsize, fontweight='bold')
        plt.legend(title="Optimizer", fontsize=legend_fontsize, title_fontsize=legend_fontsize)
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.tick_params(axis='both', which='major', labelsize=tick_fontsize) # Adjust tick label size
        plt.tight_layout()
        
        plot_path_png = Path(results_dir) / f"scalability_wall_clock_time_{device_type}.png"
        plot_path_pdf = Path(results_dir) / f"scalability_wall_clock_time_{device_type}.pdf"
        plt.savefig(plot_path_png, dpi=300)
        plt.savefig(plot_path_pdf, dpi=300)
        print(f"\nSaved plot: {plot_path_png}")
        plt.close()

    # GPU Memory Table and Plot
    df_gpu = df_results[df_results['device'] == 'cuda']
    if not df_gpu.empty:
        summary_df_gpu = df_gpu.groupby(['optimizer', 'model_config_name'])['peak_gpu_mem_allocated_mb'].max().reset_index()
        pivot_gpu_mem = summary_df_gpu.pivot_table(index="optimizer", columns="model_config_name", values="peak_gpu_mem_allocated_mb")
        pivot_gpu_mem = pivot_gpu_mem.round(2)
        pivot_gpu_mem.columns = [f"{col} (MB)" for col in pivot_gpu_mem.columns]
        pivot_gpu_mem.to_csv(Path(results_dir) / "peak_gpu_memory.csv")
        print(f"\nPeak GPU Memory (MB):\n", pivot_gpu_mem)

        plt.figure(figsize=(12, 8)) # Increased figure size
        sns.lineplot(data=df_gpu, x="model_params", y="peak_gpu_mem_allocated_mb", hue="optimizer", marker="o", markersize=8, linewidth=2, errorbar="sd")
        plt.xlabel("Number of Model Parameters", fontsize=label_fontsize)
        plt.ylabel("Peak GPU Memory Allocated (MB)", fontsize=label_fontsize)
        plt.title("Optimizer GPU Memory vs. Model Size", fontsize=title_fontsize, fontweight='bold')
        plt.legend(title="Optimizer", fontsize=legend_fontsize, title_fontsize=legend_fontsize)
        plt.xscale('log')
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.tick_params(axis='both', which='major', labelsize=tick_fontsize) # Adjust tick label size
        plt.tight_layout()
        plt.savefig(Path(results_dir) / "scalability_gpu_memory.png", dpi=300)
        plt.savefig(Path(results_dir) / "scalability_gpu_memory.pdf", dpi=300)
        print(f"\nSaved plot: {Path(results_dir) / 'scalability_gpu_memory.png'}")
        plt.close()

    # CPU Memory Table and Plot
    df_cpu_runs = df_results[df_results['device'] == 'cpu']
    if not df_cpu_runs.empty:
        summary_df_cpu = df_cpu_runs.groupby(['optimizer', 'model_config_name'])['peak_cpu_mem_usage_mb'].max().reset_index()
        pivot_cpu_mem = summary_df_cpu.pivot_table(index="optimizer", columns="model_config_name", values="peak_cpu_mem_usage_mb")
        pivot_cpu_mem = pivot_cpu_mem.round(2)
        pivot_cpu_mem.columns = [f"{col} (MB)" for col in pivot_cpu_mem.columns]
        pivot_cpu_mem.to_csv(Path(results_dir) / "peak_cpu_memory_process.csv")
        print(f"\nPeak CPU Memory (Process RSS, MB) - for CPU runs:\n", pivot_cpu_mem)

        plt.figure(figsize=(12, 8)) # Increased figure size
        sns.lineplot(data=df_cpu_runs, x="model_params", y="peak_cpu_mem_usage_mb", hue="optimizer", marker="o", markersize=8, linewidth=2, errorbar="sd")
        plt.xlabel("Number of Model Parameters", fontsize=label_fontsize)
        plt.ylabel("Peak Process CPU Memory (RSS, MB)", fontsize=label_fontsize)
        plt.title("Optimizer CPU Memory (Process) vs. Model Size (CPU Runs)", fontsize=title_fontsize, fontweight='bold')
        plt.legend(title="Optimizer", fontsize=legend_fontsize, title_fontsize=legend_fontsize)
        plt.xscale('log')
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.tick_params(axis='both', which='major', labelsize=tick_fontsize) # Adjust tick label size
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
        default=["Adagrad","RMSprop","AMSGrad","Adam","Adan","AdamP","Lion","MADGRAD","DiffGrad","PulseGrad"],
        help="List of optimizer names to benchmark."
    )
    parser.add_argument(
        "--devices",
        type=str,
        nargs="+",
        default=["cuda"],
        help="List of devices to benchmark on ('cpu', 'cuda')."
    )
    parser.add_argument("--num_steps", type=int, default=200, help="Number of optimizer steps per run.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for dummy data.")
    parser.add_argument("--input_features", type=int, default=256, help="Input features for the MLP model.")
    parser.add_argument("--num_classes", type=int, default=10, help="Output classes for the MLP model.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for optimizers.")
    parser.add_argument("--results_dir", type=str, default="./results_optimizer_benchmark", help="Directory to save tables and plots.")
    parser.add_argument(
        "--model_hidden_sizes",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512, 1024],
        help="List of hidden sizes for MLP layers for scalability testing."
    )
    parser.add_argument(
        "--model_num_layers",
        type=int,
        nargs="+",
        default=[2, 4],
        help="List of number of hidden layers for MLP for scalability testing."
    )
    args = parser.parse_args()

    model_configs = {}
    for hs in args.model_hidden_sizes:
        for nl in args.model_num_layers:
            model_configs[f"MLP_h{hs}_l{nl}"] = {"hidden_size": hs, "num_layers": nl}

    available_optimizers = []
    unique_optimizers = sorted(list(set(args.optimizers)))
    for opt_name in unique_optimizers:
        opt_name_lower = opt_name.lower()
        if opt_name_lower in ['pulsegrad', 'sgd', 'adam', 'adagrad', 'adadelta', 'rmsprop', 'amsgrad', 'diffgrad', 'adabelief', 'adamp', 'madgrad', 'adan', 'lion']:
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
