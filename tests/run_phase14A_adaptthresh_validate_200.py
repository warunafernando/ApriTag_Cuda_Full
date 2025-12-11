"""
Phase 14A full validation: CPU vs GPU adaptive threshold accuracy and speed (200 frames).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from common.config import ensure_output_dirs, load_config
from common.video import load_frames
from cpu.adaptive_threshold import cpu_adaptive_threshold

try:
    import cupy as cp
    from gpu.adaptive_threshold import gpu_adaptive_threshold
except Exception as exc:
    cp = None
    gpu_adaptive_threshold = None
    _gpu_import_error = exc
else:
    _gpu_import_error = None


def main() -> None:
    cfg = load_config()
    ensure_output_dirs(cfg)

    if gpu_adaptive_threshold is None:
        raise RuntimeError(f"GPU adaptive threshold unavailable: {_gpu_import_error}")

    video_path = cfg["input"]["video_path"]
    frames = load_frames(video_path, range(200))

    # Default parameters
    block_size = 11
    C = 2
    method = "mean"

    csv_path = Path("outputs/phase14A_adaptthresh_200.csv")
    report_path = Path("outputs/phase14A_adaptthresh_200_report.txt")

    rows = []
    cpu_times = []
    gpu_times = []
    diff_ratios = []
    mean_diffs = []
    max_diffs = []

    for idx, frame in enumerate(frames):
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # CPU path with timing
        t0 = time.perf_counter()
        bin_cpu = cpu_adaptive_threshold(gray, block_size=block_size, C=C, method=method)
        t_cpu = (time.perf_counter() - t0) * 1000.0  # ms

        # GPU path with timing
        gray_gpu = cp.asarray(gray, dtype=cp.uint8)
        # Sync before timing
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        bin_gpu = gpu_adaptive_threshold(gray_gpu, block_size=block_size, C=float(C), method=method)
        # Sync after GPU work
        cp.cuda.Stream.null.synchronize()
        t_gpu = (time.perf_counter() - t0) * 1000.0  # ms

        # Download and compare
        bin_gpu_cpu = cp.asnumpy(bin_gpu)

        # Compute accuracy metrics
        diff = np.abs(bin_cpu.astype(np.int16) - bin_gpu_cpu.astype(np.int16))
        mean_abs_diff = float(np.mean(diff))
        max_abs_diff = int(np.max(diff))
        diff_pixels = np.count_nonzero(bin_cpu != bin_gpu_cpu)
        diff_pixel_ratio = float(diff_pixels) / bin_cpu.size

        cpu_times.append(t_cpu)
        gpu_times.append(t_gpu)
        diff_ratios.append(diff_pixel_ratio)
        mean_diffs.append(mean_abs_diff)
        max_diffs.append(max_abs_diff)

        rows.append([idx, t_cpu, t_gpu, mean_abs_diff, max_abs_diff, diff_pixel_ratio])

    # Write CSV
    import csv
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_index",
            "t_cpu_adapt_ms",
            "t_gpu_adapt_ms",
            "mean_abs_diff",
            "max_abs_diff",
            "diff_pixel_ratio",
        ])
        writer.writerows(rows)

    # Compute aggregate metrics
    mean_cpu_ms = float(np.mean(cpu_times))
    mean_gpu_ms = float(np.mean(gpu_times))
    speedup = mean_cpu_ms / mean_gpu_ms if mean_gpu_ms > 0 else 0.0

    mean_diff_ratio = float(np.mean(diff_ratios))
    max_diff_ratio = float(np.max(diff_ratios))
    mean_mean_diff = float(np.mean(mean_diffs))
    max_max_diff = int(np.max(max_diffs))

    # Pass criteria
    accuracy_pass = (
        mean_diff_ratio <= 0.01
        and max_diff_ratio <= 0.02
        and mean_mean_diff <= 5.0
        and max_max_diff <= 50
    )

    speed_pass = mean_gpu_ms < mean_cpu_ms
    speed_target = mean_gpu_ms <= 0.5 * mean_cpu_ms  # 2x faster

    # Write report
    report_lines = [
        "Phase 14A Full Validation: CPU vs GPU Adaptive Threshold (200 frames)",
        "=" * 70,
        "",
        f"frames_processed: {len(frames)}",
        f"block_size: {block_size}",
        f"C: {C}",
        f"method: {method}",
        "",
        "Performance:",
        f"  mean_cpu_adapt_ms: {mean_cpu_ms:.4f}",
        f"  mean_gpu_adapt_ms: {mean_gpu_ms:.4f}",
        f"  speedup_adapt: {speedup:.2f}x",
        "",
        "Accuracy:",
        f"  mean_diff_pixel_ratio: {mean_diff_ratio:.6f}",
        f"  max_diff_pixel_ratio: {max_diff_ratio:.6f}",
        f"  mean_mean_abs_diff: {mean_mean_diff:.4f}",
        f"  max_max_abs_diff: {max_max_diff}",
        "",
        "Pass Criteria:",
        f"  Accuracy: {'PASS' if accuracy_pass else 'FAIL'}",
        f"    - mean_diff_pixel_ratio <= 0.01: {mean_diff_ratio:.6f} {'[PASS]' if mean_diff_ratio <= 0.01 else '[FAIL]'}",
        f"    - max_diff_pixel_ratio <= 0.02: {max_diff_ratio:.6f} {'[PASS]' if max_diff_ratio <= 0.02 else '[FAIL]'}",
        f"    - mean_mean_abs_diff <= 5.0: {mean_mean_diff:.4f} {'[PASS]' if mean_mean_diff <= 5.0 else '[FAIL]'}",
        f"    - max_max_abs_diff <= 50: {max_max_diff} {'[PASS]' if max_max_diff <= 50 else '[FAIL]'}",
        "",
        f"  Speed: {'PASS' if speed_pass else 'FAIL'}",
        f"    - GPU faster than CPU: {speedup:.2f}x {'[PASS]' if speed_pass else '[FAIL]'}",
        f"    - Target (2x faster): {'PASS' if speed_target else 'FAIL'} {'[PASS]' if speed_target else '[FAIL]'}",
        "",
        f"Overall: {'PASS' if (accuracy_pass and speed_pass) else 'FAIL'}",
    ]

    # Find worst frames if any failed
    if not accuracy_pass:
        worst_ratio_idx = int(np.argmax(diff_ratios))
        worst_diff_idx = int(np.argmax(mean_diffs))
        report_lines.extend([
            "",
            "Worst frames (accuracy):",
            f"  Highest diff_ratio: frame {worst_ratio_idx} (ratio={diff_ratios[worst_ratio_idx]:.6f})",
            f"  Highest mean_diff: frame {worst_diff_idx} (diff={mean_diffs[worst_diff_idx]:.4f})",
        ])

    report_path.write_text("\n".join(report_lines))
    print(f"Phase 14A validation complete: {csv_path}")
    print(f"Report: {report_path}")
    print(f"Overall: {'PASS' if (accuracy_pass and speed_pass) else 'FAIL'}")
    print(f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()

