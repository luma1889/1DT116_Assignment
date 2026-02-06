import subprocess
import re
import os
import matplotlib.pyplot as plt
import numpy as np

# Configuration
SCENARIOS = ["hugeScenario.xml", "scenario_box.xml", "scenario.xml"]
EXECUTABLE = "./demo/demo"
MAX_THREADS = 16
RUNS = 15

def run_and_get_time(flag, threads, current_scenario):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["PTHREAD_NUM_THREADS"] = str(threads)

    total = 0.0
    valid = 0
    for _ in range(RUNS):
        result = subprocess.run(
            [EXECUTABLE, "--timing-mode", flag, current_scenario],
            capture_output=True, text=True, env=env
        )
        match = re.search(r"Target time:\s*([0-9.]+)\s*milliseconds", result.stdout)
        if match:
            total += float(match.group(1))
            valid += 1
    return total / valid if valid > 0 else None

# ---- Main Loop for Scenarios ----
for scenario_file in SCENARIOS:
    scenario_name = scenario_file.replace(".xml", "")
    print(f"\n{'='*40}")
    print(f"STARTING BENCHMARK: {scenario_file}")
    print(f"{'='*40}")

    # Reset results for this specific scenario
    results = {"SEQ": [1.0]*MAX_THREADS, "OMP": [], "PTHREAD": [], "SIMD": []}
    thread_counts = list(range(1, MAX_THREADS + 1))

    # 1. Run SEQ baseline for THIS scenario
    print(f"Running SEQ baseline for {scenario_name}...")
    T_seq = run_and_get_time("--seq", 1, scenario_file)
    if T_seq is None: 
        print(f"❌ Error: Could not run SEQ for {scenario_file}. Skipping.")
        continue
    print(f"SEQ Base Time: {T_seq:.2f} ms")

    # 2. Benchmarking Loop
    for impl, flag in [("OMP", "--omp"), ("PTHREAD", "--pthread"), ("SIMD", "--simd")]:
        print(f"Benchmarking {impl}...")
        for t in thread_counts:
            print(f"  Threads: {t}", end="\r")
            T = run_and_get_time(flag, t, scenario_file)
            results[impl].append(T_seq / T if T else 0)
        print(f"\nFinished {impl}.")

    # 3. Amdahl Math (Based on OMP)
    valid_omp = [(t, s) for t, s in zip(thread_counts, results["OMP"]) if s > 0 and t > 1]
    if valid_omp:
        best_t, best_s = max(valid_omp, key=lambda x: x[1])
        p_est = (1 - 1 / best_s) / (1 - 1 / (best_t if best_t > 1 else 2))
        p = max(0.0, min(0.99, p_est)) 
    else:
        p = 0.0 

    N = np.array(thread_counts)
    amdahl = 1 / ((1 - p) + p / N)

    # 4. Plotting
    plt.figure(figsize=(12, 8), dpi=150)
    plt.plot(thread_counts, results["SEQ"], marker="o", label="SEQ (Scalar Base)")
    plt.plot(thread_counts, results["OMP"], marker="o", label="OMP (Scalar Parallel)")
    plt.plot(thread_counts, results["PTHREAD"], marker="o", label="PTHREAD (Scalar Parallel)")
    plt.plot(thread_counts, results["SIMD"], marker="s", label="SIMD + OMP (Vector Parallel)", color="purple")
    plt.plot(thread_counts, amdahl, "--", label=f"Amdahl OMP (p={p:.2f})", color='tab:red')

    plt.xlabel("Number of Threads")
    plt.ylabel("Speedup (relative to SEQ)")
    plt.title(f"Speedup Comparison: {scenario_file}\n(Baseline SEQ: {T_seq:.2f}ms)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.xticks(thread_counts)

    # Adjust limits dynamically
    max_val = max([max(results[k]) for k in results if results[k]])
    plt.ylim(0, max_val * 1.2)
    plt.xlim(1, MAX_THREADS)

    # Save with required naming convention
    save_name = f"ass2_{scenario_name}.png"
    plt.savefig(save_name, bbox_inches="tight")
    plt.close() # Important: Close plot to free memory for next scenario
    
    print(f"✅ Success! Plot saved as: {save_name}")

print(f"\n{'='*40}")
print("ALL BENCHMARKS COMPLETE")
print(f"{'='*40}")