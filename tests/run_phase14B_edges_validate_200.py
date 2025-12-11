"""
Phase 14B full validation: CPU vs GPU Canny edge detection accuracy and speed (200 frames).
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
from cpu.edges import cpu_canny_edges

try:
    import cupy as cp
    from gpu.edges import gpu_canny_edges
except Exception as exc:
    cp = None
    gpu_canny_edges = None
    _gpu_import_error = exc
else:
    _gpu_import_error = None


def main() -> None:
    cfg = load_config()
    ensure_output_dirs(cfg)

    if gpu_canny_edges is None:
        raise RuntimeError(f"GPU Canny unavailable: {_gpu_import_error}")

    video_path = cfg["input"]["video_path"]
    frames = load_frames(video_path, range(200))

    # Default parameters
    low_thresh = 35
    high_thresh = 110
    aperture_size = 3
    use_l2_gradient = True

    csv_path = Path("outputs/phase14B_edges_200.csv")
    report_path = Path("outputs/phase14B_edges_200_report.txt")

    rows = []
    cpu_times = []
    gpu_times = []
    edge_count_ratios = []
    match_ratios = []
    cpu_only_ratios = []
    gpu_only_ratios = []

    for idx, frame in enumerate(frames):
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # CPU path with timing
        t0 = time.perf_counter()
        edges_cpu = cpu_canny_edges(
            gray,
            low_thresh=low_thresh,
            high_thresh=high_thresh,
            aperture_size=aperture_size,
            use_l2_gradient=use_l2_gradient,
        )
        t_cpu = (time.perf_counter() - t0) * 1000.0  # ms

        # GPU path with timing
        gray_gpu = cp.asarray(gray, dtype=cp.uint8)
        # Sync before timing
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        edges_gpu = gpu_canny_edges(
            gray_gpu,
            low_thresh=float(low_thresh),
            high_thresh=float(high_thresh),
            aperture_size=aperture_size,
            use_l2_gradient=use_l2_gradient,
        )
        # Sync after GPU work
        cp.cuda.Stream.null.synchronize()
        t_gpu = (time.perf_counter() - t0) * 1000.0  # ms

        # Download and compare
        edges_gpu_cpu = cp.asnumpy(edges_gpu)

        # Compute accuracy metrics
        edge_count_cpu = int(np.count_nonzero(edges_cpu == 255))
        edge_count_gpu = int(np.count_nonzero(edges_gpu_cpu == 255))
        edge_count_ratio = float(edge_count_gpu / edge_count_cpu) if edge_count_cpu > 0 else 0.0

        match_mask = (edges_cpu == edges_gpu_cpu)
        match_ratio = float(np.count_nonzero(match_mask)) / edges_cpu.size

        cpu_only = (edges_cpu == 255) & (edges_gpu_cpu == 0)
        gpu_only = (edges_cpu == 0) & (edges_gpu_cpu == 255)
        cpu_only_ratio = float(np.count_nonzero(cpu_only)) / edges_cpu.size
        gpu_only_ratio = float(np.count_nonzero(gpu_only)) / edges_cpu.size

        cpu_times.append(t_cpu)
        gpu_times.append(t_gpu)
        edge_count_ratios.append(edge_count_ratio)
        match_ratios.append(match_ratio)
        cpu_only_ratios.append(cpu_only_ratio)
        gpu_only_ratios.append(gpu_only_ratio)

        rows.append([
            idx,
            edge_count_cpu,
            edge_count_gpu,
            edge_count_ratio,
            match_ratio,
            cpu_only_ratio,
            gpu_only_ratio,
            t_cpu,
            t_gpu,
        ])

    # Write CSV
    import csv
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_index",
            "edge_count_cpu",
            "edge_count_gpu",
            "edge_count_ratio",
            "match_ratio",
            "cpu_only_ratio",
            "gpu_only_ratio",
            "t_cpu_edges_ms",
            "t_gpu_edges_ms",
        ])
        writer.writerows(rows)

    # Compute aggregate metrics
    mean_cpu_ms = float(np.mean(cpu_times))
    mean_gpu_ms = float(np.mean(gpu_times))
    speedup = mean_cpu_ms / mean_gpu_ms if mean_gpu_ms > 0 else 0.0

    mean_edge_count_ratio = float(np.mean(edge_count_ratios))
    min_edge_count_ratio = float(np.min(edge_count_ratios))
    max_edge_count_ratio = float(np.max(edge_count_ratios))

    mean_match_ratio = float(np.mean(match_ratios))
    min_match_ratio = float(np.min(match_ratios))

    mean_cpu_only_ratio = float(np.mean(cpu_only_ratios))
    max_cpu_only_ratio = float(np.max(cpu_only_ratios))

    mean_gpu_only_ratio = float(np.mean(gpu_only_ratios))
    max_gpu_only_ratio = float(np.max(gpu_only_ratios))

    # Pass criteria
    accuracy_pass = (
        0.8 <= mean_edge_count_ratio <= 1.2
        and mean_match_ratio >= 0.85
        and max_cpu_only_ratio <= 0.15
        and max_gpu_only_ratio <= 0.15
    )

    speed_pass = mean_gpu_ms < mean_cpu_ms
    speed_target = speedup >= 2.0

    # Write report
    report_lines = [
        "Phase 14B Full Validation: CPU vs GPU Canny Edge Detection (200 frames)",
        "=" * 70,
        "",
        f"frames_processed: {len(frames)}",
        f"low_thresh: {low_thresh}",
        f"high_thresh: {high_thresh}",
        f"aperture_size: {aperture_size}",
        f"use_l2_gradient: {use_l2_gradient}",
        "",
        "Accuracy:",
        f"  mean_edge_count_ratio: {mean_edge_count_ratio:.6f}",
        f"  min_edge_count_ratio: {min_edge_count_ratio:.6f}",
        f"  max_edge_count_ratio: {max_edge_count_ratio:.6f}",
        f"  mean_match_ratio: {mean_match_ratio:.6f}",
        f"  min_match_ratio: {min_match_ratio:.6f}",
        f"  mean_cpu_only_ratio: {mean_cpu_only_ratio:.6f}",
        f"  max_cpu_only_ratio: {max_cpu_only_ratio:.6f}",
        f"  mean_gpu_only_ratio: {mean_gpu_only_ratio:.6f}",
        f"  max_gpu_only_ratio: {max_gpu_only_ratio:.6f}",
        "",
        "Performance:",
        f"  mean_cpu_edges_ms: {mean_cpu_ms:.4f}",
        f"  mean_gpu_edges_ms: {mean_gpu_ms:.4f}",
        f"  speedup_edges: {speedup:.2f}x",
        "",
        "Pass Criteria:",
        f"  Accuracy: {'PASS' if accuracy_pass else 'FAIL'}",
        f"    - 0.8 <= mean_edge_count_ratio <= 1.2: {mean_edge_count_ratio:.6f} {'[PASS]' if 0.8 <= mean_edge_count_ratio <= 1.2 else '[FAIL]'}",
        f"    - mean_match_ratio >= 0.85: {mean_match_ratio:.6f} {'[PASS]' if mean_match_ratio >= 0.85 else '[FAIL]'}",
        f"    - max_cpu_only_ratio <= 0.15: {max_cpu_only_ratio:.6f} {'[PASS]' if max_cpu_only_ratio <= 0.15 else '[FAIL]'}",
        f"    - max_gpu_only_ratio <= 0.15: {max_gpu_only_ratio:.6f} {'[PASS]' if max_gpu_only_ratio <= 0.15 else '[FAIL]'}",
        "",
        f"  Speed: {'PASS' if speed_pass else 'FAIL'}",
        f"    - GPU faster than CPU: {speedup:.2f}x {'[PASS]' if speed_pass else '[FAIL]'}",
        f"    - Target (2x faster): {'PASS' if speed_target else 'FAIL'} {'[PASS]' if speed_target else '[FAIL]'}",
        "",
        f"Overall: {'PASS' if (accuracy_pass and speed_pass) else 'FAIL'}",
    ]

    report_path.write_text("\n".join(report_lines))
    print(f"Phase 14B validation complete: {csv_path}")
    print(f"Report: {report_path}")
    print(f"Overall: {'PASS' if (accuracy_pass and speed_pass) else 'FAIL'}")
    print(f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()

