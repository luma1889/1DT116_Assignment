import subprocess
import re
import os
import matplotlib.pyplot as plt
import numpy as np

SCENARIO = "scenario_box.xml"
EXECUTABLE = "./demo/demo"
MAX_THREADS = 16
RUNS = 1

def run_and_get_time(flag, threads):
    env = os.environ.copy()
    # Ensure BOTH parallel frameworks get the thread count
    env["OMP_NUM_THREADS"] = str(threads)
    env["PTHREAD_NUM_THREADS"] = str(threads)

    total = 0.0
    valid = 0
    for _ in range(RUNS):
        result = subprocess.run(
            [EXECUTABLE, "--timing-mode", flag, SCENARIO],
            capture_output=True, text=True, env=env
        )
        match = re.search(r"Target time:\s*([0-9.]+)\s*milliseconds", result.stdout)
        if match:
            total += float(match.group(1))
            valid += 1
    return total / valid if valid > 0 else None

# ---- Run SEQ baseline ----
print("Running SEQ baseline...")
T_seq = run_and_get_time("--seq", 1)
if T_seq is None: 
    raise RuntimeError("Failed to get SEQ baseline. Is your program compiled?")
print(f"SEQ Time: {T_seq:.2f} ms\n")

thread_counts = list(range(1, MAX_THREADS + 1))
results = {"SEQ": [1.0]*MAX_THREADS, "OMP": [], "PTHREAD": []}

# ---- Benchmarking Loop with Detailed Output ----
for impl, flag in [("OMP", "--omp"), ("PTHREAD", "--pthread")]:
    print(f"Benchmarking {impl}:")
    for t in thread_counts:
        print(f"  Threads: {t}") # This shows you the progress
        T = run_and_get_time(flag, t)
        results[impl].append(T_seq / T if T else None)
    print(f"Finished {impl}.\n")

# ---- Amdahl Math ----
valid_omp = [(t, s) for t, s in zip(thread_counts, results["OMP"]) if s is not None and t > 1]
if valid_omp:
    # Estimate 'p' (clamped for realistic curve)
    realistic_points = [(t, s) for t, s in valid_omp if s <= t]
    if realistic_points:
        best_t, best_s = max(realistic_points, key=lambda x: x[1])
    else:
        best_t, best_s = max(valid_omp, key=lambda x: x[1])
    
    p_est = (1 - 1 / best_s) / (1 - 1 / best_t)
    p = max(0.0, min(0.95, p_est)) 
else:
    p = 0.85 

N = np.array(thread_counts)
amdahl = 1 / ((1 - p) + p / N)

# ---- Plotting ----
plt.figure(figsize=(10, 7), dpi=150)
plt.plot(thread_counts, results["SEQ"], marker="o", label="SEQ")
plt.plot(thread_counts, results["OMP"], marker="o", label="OMP")
plt.plot(thread_counts, results["PTHREAD"], marker="o", label="PTHREAD")
plt.plot(thread_counts, amdahl, "--", label=f"Amdahl (p={p:.2f})", color='tab:red')

plt.xlabel("Number of Threads")
plt.ylabel("Speedup")
plt.title("Speedup vs Number of Threads")
plt.legend()
plt.grid(True)
plt.xticks(thread_counts)

# Constraints for the graph
plt.xlim(1, MAX_THREADS)  # Start at thread 1
plt.ylim(0, 10)  # Cap at 10 for visibility

plt.savefig("speedup_plot.png", bbox_inches="tight")
print(f"\n✅ Parallel fraction p ≈ {p:.2f}")
print("✅ Speedup plot saved as speedup_plot.png")