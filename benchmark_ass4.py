import subprocess
import re
import os
import matplotlib.pyplot as plt
import numpy as np

# ================= CONFIGURATION =================
SCENARIOS = ["scenario_box.xml", "hugeScenario.xml"]
# SCENARIOS = ["scenario.xml"]

EXECUTABLE = "./demo/demo"
MAX_THREADS = 16
RUNS = 5  # Multiple runs to smooth out background noise
# =================================================

def run_simulation(flag, threads, scenario):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    
    # Using --t (Timing Mode) and --max-steps=200 for consistent results
    result = subprocess.run(
        [EXECUTABLE, "--t", "--max-steps=200", flag, scenario],
        capture_output=True,
        text=True,
        env=env
    )
    
    # We want the 'Target' section (the second block of text)
    parts = result.stdout.split("Running target version...")
    return parts[1] if len(parts) > 1 else result.stdout

def parse_timings(output):
    """Parses all timing data. Returns 0.0 for missing keys to avoid KeyError."""
    data = {}
    
    # Standard Target time
    t_match = re.search(r"Target time:\s*([0-9.]+)", output)
    data["total_target"] = float(t_match.group(1)) if t_match else 0.0

    # CPU Heatmap
    cpu_match = re.search(r"Average PURE CPU Heatmap time:\s*([0-9.]+)", output)
    data["cpu_hmap"] = float(cpu_match.group(1)) if cpu_match else 0.0

    # GPU Breakdown
    gpu_keys = ["fade", "inc", "scale", "blur", "total"]
    for key in gpu_keys:
        pattern = rf"Average (?:TOTAL )?GPU {key.capitalize()}:\s*([0-9.]+)"
        if key == "total":
            pattern = r"Average TOTAL GPU:\s*([0-9.]+)"
        m = re.search(pattern, output)
        data[f"gpu_{key}"] = float(m.group(1)) if m else 0.0
            
    return data

def benchmark_impl(flag, threads, scenario):
    """Averages metrics over multiple runs for a single configuration."""
    accumulated = {}
    count = 0
    for _ in range(RUNS):
        out = run_simulation(flag, threads, scenario)
        run_data = parse_timings(out)
        if not run_data: continue
        for k, v in run_data.items():
            accumulated[k] = accumulated.get(k, 0) + v
        count += 1
    return {k: v / count for k, v in accumulated.items()} if count > 0 else None

# ======================= MAIN =======================
for scenario_file in SCENARIOS:
    scenario_name = scenario_file.replace(".xml", "")
    print(f"\nBenchmarking Scenario: {scenario_file}")

    # 1. Sequential Baseline
    print(f"  Calculating SEQ Baseline ({RUNS} runs)...")
    seq_avg = benchmark_impl("--seq", 1, scenario_file)
    t_seq_base = seq_avg["total_target"]

    # 2. Scaling Sweep (1 to 16 threads)
    thread_results = []
    thread_counts = list(range(1, MAX_THREADS + 1))
    
    print(f"  Scaling REG Implementation (Threads 1-{MAX_THREADS})...")
    for t in thread_counts:
        print(f"    Testing {t} threads...", end="\r")
        res = benchmark_impl("--reg", t, scenario_file)
        thread_results.append(res)
    print("\n  ✅ Scaling data collected.")

    # 3. Find the Best performing thread count for the other graphs
    best_res = min(thread_results, key=lambda x: x["total_target"])
    best_t_idx = thread_results.index(best_res)
    best_t = thread_counts[best_t_idx]

    # ================= PLOTTING =================

    # Graph 1: Scaling Speedup (Total Simulation)
    plt.figure(figsize=(10, 7), dpi=150)
    speedups = [t_seq_base / r["total_target"] for r in thread_results]
    
    plt.plot(thread_counts, speedups, marker='o', color='#0072B8', label="REG (Heterogeneous)")
    plt.axhline(y=1.0, color='#FF6F20', linestyle='--', label="SEQ (Baseline)")
    
    plt.xlabel("Number of Threads")
    plt.ylabel("Speedup (relative to SEQ)")
    plt.title(f"Simulation Scaling: {scenario_file}\n(Best time: {best_t} Threads at {best_res['total_target']:.2f}ms)")
    plt.grid(True, alpha=0.3)
    plt.xticks(thread_counts)
    plt.legend()
    plt.savefig(f"ass4_scaling_{scenario_name}.png")
    plt.close()

    # Graph 2: Component Speedup (using best thread result)
    plt.figure(figsize=(10, 6), dpi=150)
    h_labels = ['CPU Heatmap (SEQ)', f'GPU Heatmap (REG at {best_t} Threads)']
    h_times = [seq_avg['cpu_hmap'], best_res['gpu_total']]
    
    bars = plt.bar(h_labels, h_times, color=['#FF6F20', '#0072B8'], width=0.4)
    plt.ylabel("Execution Time (ms)")
    plt.title(f"Component Benchmark: Heatmap Speedup\nScenario: {scenario_file}")
    
    h_speedup = seq_avg['cpu_hmap'] / best_res['gpu_total']
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                 f"{bar.get_height():.3f} ms", ha='center', va='bottom', fontweight='bold')
    
    plt.text(0.5, seq_avg['cpu_hmap']*0.5, f"Heatmap Speedup: {h_speedup:.1f}x", 
             ha='center', fontsize=12, fontweight='bold', bbox=dict(facecolor='#FFD700', alpha=0.5))
    
    plt.savefig(f"ass4_hmap_speedup_{scenario_name}.png")
    plt.close()

    # Graph 3: GPU Breakdown (using best thread result)
    plt.figure(figsize=(10, 6), dpi=150)
    b_labels = ['Fade', 'Increment', 'Scale', 'Blur']
    b_times = [best_res['gpu_fade'], best_res['gpu_inc'], best_res['gpu_scale'], best_res['gpu_blur']]
    
    plt.bar(b_labels, b_times, color=['#0072B8', '#A3D5E0', '#FFD700', '#FF6F20'])
    plt.ylabel("Execution Time (ms)")
    plt.title(f"GPU Heatmap Kernel Breakdown\nScenario: {scenario_file}\n(Best Run: {best_t} Threads)")
    
    for i, v in enumerate(b_times):
        plt.text(i, v, f"{v:.4f} ms", ha='center', va='bottom', fontweight='bold')

    plt.savefig(f"ass4_gpu_breakdown_{scenario_name}.png")
    plt.close()

    print(f"✅ Finished {scenario_file}. Best Configuration: {best_t} threads.")