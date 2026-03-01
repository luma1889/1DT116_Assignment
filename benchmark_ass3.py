import subprocess
import re
import os
import matplotlib.pyplot as plt
import numpy as np

# ================= CONFIGURATION =================
SCENARIOS = ["hugeScenario.xml", "scenario_box.xml", "scenario.xml"]
EXECUTABLE = "./demo/demo"
MAX_THREADS = 16
RUNS = 1  # Increase to 3+ for stable averages
# =================================================


def run_and_get_time(flag, threads, current_scenario):
    """Run executable and extract average execution time in ms."""
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    # env["PTHREAD_NUM_THREADS"] = str(threads)

    total = 0.0
    valid = 0

    for _ in range(RUNS):
        result = subprocess.run(
            [EXECUTABLE, "--timing-mode", flag, current_scenario],
            capture_output=True,
            text=True,
            env=env
        )

        match = re.search(r"Target time:\s*([0-9.]+)", result.stdout)
        if match:
            total += float(match.group(1))
            valid += 1

    return total / valid if valid > 0 else None


def plot_speedup_scaling(
    scenario_file, scenario_name, thread_counts,
    line_results, best_times, T_seq
):
    """Plot speedup scaling graph."""
    plt.figure(figsize=(12, 8), dpi=150)

    plt.plot(thread_counts, line_results["SEQ"], marker="o", label="SEQ (Baseline)")
    plt.plot(thread_counts, line_results["REG"], marker="o", label="REGION + OMP")
    # plt.plot(thread_counts, line_results["PTHREAD"], marker="o", label="PTHREAD")
    # plt.plot(thread_counts, line_results["SIMD"], marker="s", label="SIMD + OMP")

    # if best_times["CUDA"] is not None:
    #     cuda_speedup = T_seq / best_times["CUDA"]
    #     plt.axhline(
    #         y=cuda_speedup,
    #         linestyle="--",
    #         label=f"CUDA Speedup ({cuda_speedup:.2f}x)"
    #     )

    # plt.plot(
    #     thread_counts,
    #     amdahl,
    #     "--",
    #     label=f"Amdahl OMP (p={p:.2f})"
    # )

    plt.xlabel("Number of Threads")
    plt.ylabel("Speedup (relative to SEQ)")
    plt.title(f"Speedup Scaling: {scenario_file}")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.xticks(thread_counts)
    plt.xlim(1, MAX_THREADS)

    max_speedup = max(
        max(v) for v in line_results.values() if v
    )
    # if best_times["CUDA"] is not None:
    #     max_speedup = max(max_speedup, T_seq / best_times["CUDA"])

    plt.ylim(0, max_speedup * 1.1)

    plt.savefig(f"ass3_many_regions{scenario_name}.png", bbox_inches="tight")
    plt.close()


def plot_best_times_bar(scenario_file, scenario_name, best_times):
    """Plot best execution times bar chart."""
    plt.figure(figsize=(12, 7), dpi=150)

    labels = [
        "Serial",
        "REGION + OMP (BEST)",
        # "PTHREAD (Best)",
        # "SIMD (Best)",
        # "CUDA"
    ]

    final_times = [
        best_times["SEQ"],
        best_times["REG"],
        # best_times["PTHREAD"],
        # best_times["SIMD"],
        # best_times["CUDA"] if best_times["CUDA"] is not None else 0
    ]

    bars = plt.bar(labels, final_times)

    plt.ylabel("Execution Time (ms)")
    plt.title(f"Evaluation: Best Execution Times\nScenario: {scenario_file}")

    for bar in bars:
        yval = bar.get_height()
        if yval > 0:
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                yval,
                f"{yval:.2f} ms",
                ha="center",
                va="bottom",
                fontweight="bold"
            )
        else:
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                5,
                "N/A",
                ha="center",
                va="bottom",
                color="red"
            )

    plt.savefig(f"ass3_many_regions{scenario_name}_bar.png", bbox_inches="tight")
    plt.close()


# def estimate_amdahl(thread_counts, omp_speedups):
#     """Estimate Amdahl's p using best OMP scaling point."""
#     valid = [
#         (t, s)
#         for t, s in zip(thread_counts, omp_speedups)
#         if s > 0 and t > 1
#     ]

#     p = 0.0
#     if valid:
#         best_t, best_s = max(valid, key=lambda x: x[1])
#         p_est = (1 - 1 / best_s) / (1 - 1 / best_t)
#         p = max(0.0, min(0.99, p_est))

#     amdahl = 1 / ((1 - p) + p / np.array(thread_counts))
#     return amdahl, p


# ======================= MAIN =======================
for scenario_file in SCENARIOS:
    scenario_name = scenario_file.replace(".xml", "")
    thread_counts = list(range(1, MAX_THREADS + 1))

    print("\n" + "=" * 60)
    print(f"STARTING BENCHMARK: {scenario_file}")
    print("=" * 60)

    best_times = {
        "SEQ": None,
        "REG": float("inf"),
        # "PTHREAD": float("inf"),
        # "SIMD": float("inf"),
        # "CUDA": None
    }

    line_results = {
        "SEQ": [1.0] * MAX_THREADS,
        "REG": [],
        # "PTHREAD": [],
        # "SIMD": []
    }

    # ---- SEQ baseline ----
    print("Running SEQ baseline...")
    T_seq = run_and_get_time("--seq", 1, scenario_file)
    if T_seq is None:
        print("❌ SEQ failed, skipping scenario.")
        continue

    best_times["SEQ"] = T_seq
    print(f"SEQ Base Time: {T_seq:.2f} ms")

    # ---- CUDA ----
    # print("Running CUDA (Bonus)...")
    # T_cuda = run_and_get_time("--cuda", 1, scenario_file)
    # if T_cuda is not None:
    #     best_times["CUDA"] = T_cuda
    #     print(f"CUDA Time: {T_cuda:.2f} ms")

    # ---- Scaling ----
    for impl, flag in [
        ("REG", "--reg"),
        # ("PTHREAD", "--pthread"),
        # ("SIMD", "--simd")
    ]:
        print(f"Benchmarking {impl} scaling...")
        for t in thread_counts:
            T_avg = run_and_get_time(flag, t, scenario_file)
            if T_avg is not None:
                speedup = T_seq / T_avg
                line_results[impl].append(speedup)
                best_times[impl] = min(best_times[impl], T_avg)
            else:
                line_results[impl].append(0.0)

    # ---- Amdahl ----
    # amdahl, p = estimate_amdahl(thread_counts, line_results["OMP"])

    # ---- Plots ----
    plot_speedup_scaling(
        scenario_file,
        scenario_name,
        thread_counts,
        line_results,
        best_times,
        T_seq,
        # amdahl,
        # p
    )

    plot_best_times_bar(
        scenario_file,
        scenario_name,
        best_times
    )

    print(f"✅ Finished Scenario: {scenario_file}")
    print(f"   Saved: ass2_many_regions{scenario_name}.png")
    print(f"   Saved: ass2_many_regions{scenario_name}_bar.png")
