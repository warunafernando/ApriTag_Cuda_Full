"""
Phase 14A smoke test: CPU vs GPU adaptive threshold accuracy (5 frames).
"""

from __future__ import annotations

import sys
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
    frames = load_frames(video_path, range(5))

    # Default parameters (can be made configurable later)
    block_size = 11
    C = 2
    method = "mean"

    csv_path = Path("outputs/phase14A_adaptthresh_smoke5.csv")
    report_path = Path("outputs/phase14A_adaptthresh_smoke5_report.txt")

    rows = []
    all_passed = True

    for idx, frame in enumerate(frames):
        # Convert to grayscale
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # CPU path
        bin_cpu = cpu_adaptive_threshold(gray, block_size=block_size, C=C, method=method)

        # GPU path
        gray_gpu = cp.asarray(gray, dtype=cp.uint8)
        bin_gpu = gpu_adaptive_threshold(gray_gpu, block_size=block_size, C=float(C), method=method)
        bin_gpu_cpu = cp.asnumpy(bin_gpu)

        # Compute metrics
        diff = np.abs(bin_cpu.astype(np.int16) - bin_gpu_cpu.astype(np.int16))
        mean_abs_diff = float(np.mean(diff))
        max_abs_diff = int(np.max(diff))
        diff_pixels = np.count_nonzero(bin_cpu != bin_gpu_cpu)
        total_pixels = bin_cpu.size
        diff_pixel_ratio = float(diff_pixels) / total_pixels

        # Pass criteria
        passed = (
            diff_pixel_ratio <= 0.01
            and mean_abs_diff <= 5.0
            and max_abs_diff <= 50
        )

        if not passed:
            all_passed = False

        rows.append([idx, mean_abs_diff, max_abs_diff, diff_pixel_ratio, passed])

    # Write CSV
    import csv
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "mean_abs_diff", "max_abs_diff", "diff_pixel_ratio", "passed"])
        writer.writerows(rows)

    # Write report
    report_lines = [
        "Phase 14A Smoke Test: CPU vs GPU Adaptive Threshold (5 frames)",
        "=" * 70,
        "",
        f"frames_processed: {len(frames)}",
        f"block_size: {block_size}",
        f"C: {C}",
        f"method: {method}",
        "",
        "Per-frame results:",
    ]

    for row in rows:
        idx, mean_diff, max_diff, ratio, passed = row
        status = "PASS" if passed else "FAIL"
        report_lines.append(
            f"  Frame {idx}: mean_diff={mean_diff:.4f}, max_diff={max_diff}, "
            f"diff_ratio={ratio:.6f} [{status}]"
        )

    report_lines.extend([
        "",
        f"Overall: {'PASS' if all_passed else 'FAIL'}",
        "",
        "Pass criteria:",
        "  - diff_pixel_ratio <= 0.01",
        "  - mean_abs_diff <= 5.0",
        "  - max_abs_diff <= 50",
    ])

    report_path.write_text("\n".join(report_lines))
    print(f"Phase 14A smoke test complete: {csv_path}")
    print(f"Report: {report_path}")
    print(f"Overall: {'PASS' if all_passed else 'FAIL'}")


if __name__ == "__main__":
    main()

