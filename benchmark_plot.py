import subprocess
import re
import os
import matplotlib.pyplot as plt
import numpy as np

# Configuration
SCENARIOS = ["hugeScenario.xml", "scenario_box.xml", "scenario.xml"]
EXECUTABLE = "./demo/demo"
MAX_THREADS = 16
RUNS = 1 # Set to 3 for reliable averages

def run_and_get_time(flag, threads, current_scenario):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["PTHREAD_NUM_THREADS"] = str(threads)

    total = 0.0
    valid = 0
    for _ in range(RUNS):
        try:
            result = subprocess.run(
                [EXECUTABLE, "--timing-mode", flag, current_scenario],
                capture_output=True, text=True, env=env
            )
            match = re.findall(r"Target time:\s*([0-9.]+)\s*milliseconds", result.stdout)
            if match:
                total += float(match.group(1))
                valid += 1
        except Exception as e:
            print(f"Error: {e}")
            return None
    return total / valid if valid > 0 else None

# ---- Main Loop for Scenarios ----
for scenario_file in SCENARIOS:
    scenario_name = scenario_file.replace(".xml", "")
    print(f"\n{'='*60}")
    print(f"STARTING BENCHMARK: {scenario_file}")
    print(f"{'='*60}")

    # --- EVERYTHING BELOW IS NOW PROPERLY INDENTED INSIDE THE LOOP ---
    best_times = {
        "SEQ": 0.0,
        "OMP": float('inf'),
        "PTHREAD": float('inf'),
        "SIMD": float('inf'),
        "CUDA": 0.0
    }

    # Results for the Scaling Line Plot
    line_results = {"SEQ": [1.0]*MAX_THREADS, "OMP": [], "PTHREAD": [], "SIMD": []}
    thread_counts = list(range(1, MAX_THREADS + 1))

    # 1. Run SEQ baseline
    print(f"Running SEQ baseline...")
    T_seq = run_and_get_time("--seq", 1, scenario_file)
    if T_seq is None: 
        print(f"Skipping {scenario_file} due to SEQ failure.")
        continue
    best_times["SEQ"] = T_seq
    print(f"SEQ Base Time: {T_seq:.2f} ms")

    # 2. Run CUDA (Bonus)
    print(f"Running CUDA (Bonus Point)...")
    T_cuda = run_and_get_time("--cuda", 1, scenario_file)
    if T_cuda:
        best_times["CUDA"] = T_cuda
        print(f"CUDA Time: {T_cuda:.2f} ms")

    # 3. Benchmarking Scaling Loop (OMP, PTHREAD, SIMD)
    for impl, flag in [("OMP", "--omp"), ("PTHREAD", "--pthread"), ("SIMD", "--simd")]:
        print(f"Benchmarking {impl} scaling...")
        for t in thread_counts:
            print(f"  Threads: {t}", end="\r")
            T_avg = run_and_get_time(flag, t, scenario_file)
            
            if T_avg:
                # Store speedup for line plot
                line_results[impl].append(T_seq / T_avg)
                # Track best time for bar plot
                if T_avg < best_times[impl]:
                    best_times[impl] = T_avg
            else:
                line_results[impl].append(0)
        print(f"\nFinished {impl}.")

    # 4. Amdahl Math (Based on OMP scaling)
    valid_omp = [(t, s) for t, s in zip(thread_counts, line_results["OMP"]) if s > 0 and t > 1]
    p = 0.0
    if valid_omp:
        best_t, best_s = max(valid_omp, key=lambda x: x[1])
        p_est = (1 - 1 / best_s) / (1 - 1 / best_t)
        p = max(0.0, min(0.99, p_est)) 
    amdahl = 1 / ((1 - p) + p / np.array(thread_counts))

    # --- PLOT 1: SPEEDUP SCALING (Line Graph) ---
    plt.figure(figsize=(12, 8), dpi=150)
    
    # Ensure X-axis starts exactly at 1 and touches the Y-axis
    plt.xlim(1, MAX_THREADS)
    plt.xticks(thread_counts)
    
    # Baseline check for 0.00ms (avoids DivisionByZero)
    Safe_T_seq = T_seq if T_seq > 0.1 else 1.0 

    plt.plot(thread_counts, [1.0]*MAX_THREADS, label="SEQ (Baseline)", color='gray', linestyle=':')
    plt.plot(thread_counts, line_results["OMP"], marker="o", label="OMP")
    plt.plot(thread_counts, line_results["PTHREAD"], marker="o", label="PTHREAD")
    plt.plot(thread_counts, line_results["SIMD"], marker="s", label="SIMD + OMP", color="purple")
    plt.plot(thread_counts, amdahl, "--", label=f"Amdahl OMP (p={p:.2f})", color='tab:red')
    # Add CUDA as a horizontal dashed line
    if best_times["CUDA"] > 0:
        cuda_speedup = Safe_T_seq / best_times["CUDA"]
        plt.axhline(y=cuda_speedup, color='blue', linestyle='--', label=f"CUDA ({cuda_speedup:.2f}x)")

    plt.xlabel("Number of Threads")
    plt.ylabel("Speedup (relative to SEQ)")
    plt.title(f"Speedup Comparison: {scenario_file}\n(Baseline SEQ: {T_seq:.2f}ms)")
    plt.legend(loc='upper left')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.ylim(bottom=0) # Ensure Y starts at 0

    # Force X-axis start point
    plt.gca().set_xbound(lower=1, upper=MAX_THREADS)

    # Dynamic Y-limit
    all_speedups = [s for sublist in line_results.values() for s in sublist]
    if best_times["CUDA"] > 0: all_speedups.append(T_seq / best_times["CUDA"])
    max_speedup = max(all_speedups) if all_speedups else 1
    plt.ylim(0, max_speedup * 1.1)

    plt.savefig(f"ass2_{scenario_name}.png", bbox_inches="tight")
    plt.close()

    # --- PLOT 2: EXECUTION TIME (Bar Plot - Section 3.6) ---
    plt.figure(figsize=(12, 7), dpi=150)
    labels = ['Serial', 'OMP (Best)', 'PTHREAD (Best)', 'SIMD (Best)', 'CUDA']
    final_times = [best_times["SEQ"], best_times["OMP"], best_times["PTHREAD"], best_times["SIMD"], best_times["CUDA"]]
    colors = ['#808080', '#ff9900', '#2ca02c', '#9467bd', '#1f77b4']
    bars = plt.bar(labels, final_times, color=colors)
    plt.ylabel("Execution Time (ms)")
    plt.title(f"3.6 Evaluation: Best Execution Times\nScenario: {scenario_file}")

    for bar in bars:
        yval = bar.get_height()
        if yval > 0 and yval != float('inf'):
            plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}ms', 
                     va='bottom', ha='center', fontweight='bold')
        else:
            plt.text(bar.get_x() + bar.get_width()/2, 5, 'N/A', 
                     va='bottom', ha='center', color='red')

    plt.savefig(f"ass2_{scenario_name}_bar.png", bbox_inches="tight")
    plt.close()

    print(f"✅ Finished Scenario: {scenario_file}")
    print(f"   Saved: ass2_{scenario_name}.png and ass2_{scenario_name}_bar.png")