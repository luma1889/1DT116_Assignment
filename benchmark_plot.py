import subprocess
import re
import os
import matplotlib.pyplot as plt
import numpy as np

# Configuration
SCENARIOS = ["hugeScenario.xml", "scenario_box.xml", "scenario.xml"]
EXECUTABLE = "./demo/demo"
MAX_THREADS = 16
RUNS = 20# Set to 3-5 for final submission to get smooth lines

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
            
            # findall returns a list of strings found
            matches = re.findall(r"Target time:\s*([0-9.]+)\s*milliseconds", result.stdout)
            
            if matches:
                # matches[-1] is the LAST match (the simulation, not the cuda test)
                # No .group(1) needed, findall already extracted the number
                total += float(matches[-1])
                valid += 1
            else:
                print(f"⚠️ Warning: No timing match found for {flag} in {current_scenario}")
                
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

    best_times = {
        "SEQ": 0.0,
        "OMP": float('inf'),
        "PTHREAD": float('inf'),
        "SIMD": float('inf'),
        "CUDA": 0.0
    }

    line_results = {"SEQ": [1.0]*MAX_THREADS, "OMP": [], "PTHREAD": [], "SIMD": []}
    thread_counts = list(range(1, MAX_THREADS + 1))

    # 1. Run SEQ baseline
    print(f"Running SEQ baseline...")
    T_seq = run_and_get_time("--seq", 1, scenario_file)
    if T_seq is None or T_seq == 0: 
        # If scenario_box is 0.0, we use a tiny value to avoid division by zero
        T_seq = 0.01 if T_seq == 0 else None
        
    if T_seq is None:
        print(f"Skipping {scenario_file} due to SEQ failure.")
        continue
        
    best_times["SEQ"] = T_seq
    print(f"SEQ Base Time: {T_seq:.2f} ms")

    # 2. Run CUDA (Bonus)
    # print(f"Running CUDA (Bonus Point)...")
    # T_cuda = run_and_get_time("--cuda", 1, scenario_file)
    # if T_cuda:
    #     best_times["CUDA"] = T_cuda
    #     print(f"CUDA Time: {T_cuda:.2f} ms")

    # 3. Benchmarking Scaling Loop
    for impl, flag in [("OMP", "--omp"), ("PTHREAD", "--pthread")]:
        print(f"Benchmarking {impl} scaling...")
        for t in thread_counts:
            print(f"  Threads: {t}", end="\r")
            T_avg = run_and_get_time(flag, t, scenario_file)
            
            if T_avg:
                line_results[impl].append(T_seq / T_avg)
                if T_avg < best_times[impl]:
                    best_times[impl] = T_avg
            else:
                line_results[impl].append(0)
        print(f"\nFinished {impl}.")

    # 4. Amdahl Math
    valid_omp = [(t, s) for t, s in zip(thread_counts, line_results["OMP"]) if s > 0 and t > 1]
    p = 0.0
    if valid_omp:
        best_t, best_s = max(valid_omp, key=lambda x: x[1])
        p_est = (1 - 1 / best_s) / (1 - 1 / best_t)
        p = max(0.0, min(0.99, p_est)) 
    amdahl = 1 / ((1 - p) + p / np.array(thread_counts))

    # --- PLOT 1: SPEEDUP SCALING ---
    plt.figure(figsize=(12, 8), dpi=150)
    plt.xlim(1, MAX_THREADS)
    plt.xticks(thread_counts)
    
    plt.plot(thread_counts, [1.0]*MAX_THREADS, label="SEQ (Baseline)", color='gray', linestyle=':')
    plt.plot(thread_counts, line_results["OMP"], marker="o", label="OMP")
    plt.plot(thread_counts, line_results["PTHREAD"], marker="o", label="PTHREAD")
    # plt.plot(thread_counts, line_results["SIMD"], marker="s", label="SIMD + OMP", color="purple")
    plt.plot(thread_counts, amdahl, "--", label=f"Amdahl OMP (p={p:.2f})", color='tab:red')
    
    if best_times["CUDA"] > 0:
        cuda_speedup = T_seq / best_times["CUDA"]
        plt.axhline(y=cuda_speedup, color='blue', linestyle='--', label=f"CUDA ({cuda_speedup:.2f}x)")

    plt.xlabel("Number of Threads")
    plt.ylabel("Speedup (relative to SEQ)")
    plt.title(f"Speedup Scaling: {scenario_file}\n(Baseline SEQ: {T_seq:.2f}ms)")
    plt.legend(loc='upper left')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.ylim(bottom=0)
    
    # Ensures the plot lines touch the Y-axis
    plt.gca().set_xbound(lower=1, upper=MAX_THREADS)

    save_name = f"ass1_{scenario_name}.png"
    plt.savefig(save_name, bbox_inches="tight")
    plt.close()

    # --- PLOT 2: BAR PLOT ---
    plt.figure(figsize=(12, 7), dpi=150)
    labels = ['Serial', 'OMP', 'PTHREAD', 'SIMD', 'CUDA']
    final_times = [best_times["SEQ"], best_times["OMP"], best_times["PTHREAD"], best_times["SIMD"], best_times["CUDA"]]
    
    # Replace inf with 0 for plotting
    plot_times = [0 if t == float('inf') else t for t in final_times]
    
    colors = ['#808080', '#ff9900', '#2ca02c', '#9467bd', '#1f77b4']
    bars = plt.bar(labels, plot_times, color=colors)
    plt.ylabel("Execution Time (ms)")
    plt.title(f"Best Execution Times\nScenario: {scenario_file}")

    for bar in bars:
        yval = bar.get_height()
        if yval > 0:
            plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}ms', 
                     va='bottom', ha='center', fontweight='bold')
        else:
            plt.text(bar.get_x() + bar.get_width()/2, 5, 'N/A', 
                     va='bottom', ha='center', color='red')

    plt.savefig(f"ass1_{scenario_name}_bar.png", bbox_inches="tight")
    plt.close()

    print(f"✅ Finished Scenario: {scenario_file}")

print(f"\n{'='*60}\nALL BENCHMARKS COMPLETE\n{'='*60}")