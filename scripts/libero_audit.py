import torch
import numpy as np
import time
import gc
from models.SmolLM2 import SmolVLA, apply_compression


def run_smollm2_vla_audit():
    # 1. HARDWARE INITIALIZATION
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    print(f"🚀 Initializing SmolVLA Audit on: {device}")

    # 2. LOAD & COMPRESS THE 97M MODEL [cite: 907]
    vla = SmolVLA().to(device).to(torch.bfloat16)

    # Generate dummy calibration data for CKA-driven pruning [cite: 894]
    dummy_img = torch.randn(1, 3, 224, 224).to(device).to(torch.bfloat16)
    dummy_txt = vla.tokenizer("pick up the black bowl", return_tensors="pt").input_ids.to(device)
    dummy_st = torch.randn(1, 32).to(device).to(torch.bfloat16)

    # Apply 57% parameter reduction logic [cite: 907]
    apply_compression(vla, dummy_img, dummy_txt, dummy_st)
    vla.eval()

    # Final Parameter Verification [cite: 907]
    vla.audit("OPTIMIZED 97M CORE")

    # 3. DEFINE LIBERO SUITES
    libero_suites = {
        "LIBERO_SPATIAL": [
            "pick up the black bowl and place it on the plate",
            "pick up the black bowl next to the ramekin",
            "pick up the black bowl on the cookie box"
        ],
        "LIBERO_OBJECT": [
            "pick up the alphabet soup and put it in the basket",
            "pick up the tomato soup and put it in the basket"
        ]
    }

    print(f"\n[!] INITIATING PERFORMANCE BENCHMARKING")

    suite_metrics = {}

    for suite, tasks in libero_suites.items():
        print(f"\n" + "=" * 60 + f"\nSUITE: {suite}\n" + "=" * 60)
        suite_latencies = []

        for task in tasks:
            # Tokenize the specific instruction [cite: 883]
            text_ids = vla.tokenizer(task, return_tensors="pt").input_ids.to(device)
            task_times = []

            # Run 10 iterations per task to get a stable average [cite: 914]
            with torch.no_grad():
                for i in range(10):
                    # Prepare input tensors [cite: 875, 876]
                    img = torch.randn(1, 3, 224, 224).to(device).to(torch.bfloat16)
                    state = torch.randn(1, 32).to(device).to(torch.bfloat16)

                    # Start High-Resolution Timer
                    t0 = time.perf_counter()
                    _ = vla(img, text_ids, state)
                    t1 = time.perf_counter()

                    # Log time in milliseconds (skip index 0 to avoid cold-start bias)
                    if i > 0:
                        task_times.append((t1 - t0) * 1000)

            avg_task_lat = np.mean(task_times)
            avg_task_hz = 1000 / avg_task_lat  # Frequency = 1 / Latency
            suite_latencies.append(avg_task_lat)

            print(f"Task: {task[:35]:<35} | Latency: {avg_task_lat:6.2f}ms | Freq: {avg_task_hz:6.2f}Hz")

        suite_metrics[suite] = {
            "lat": np.mean(suite_latencies),
            "hz": 1000 / np.mean(suite_latencies)
        }

    # 4. FINAL AGGREGATED REPORT [cite: 919, 920]
    print("\n" + "#" * 60)
    print("           SMOL-VLA PERFORMANCE TOTALS")
    print("#" * 60)
    print(f"{'SUITE':<20} | {'AVG LATENCY':<15} | {'AVG FREQUENCY':<15}")
    print("-" * 60)
    for suite, m in suite_metrics.items():
        print(f"{suite:<20} | {m['lat']:>12.2f} ms | {m['hz']:>12.2f} Hz")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    run_smollm2_vla_audit()