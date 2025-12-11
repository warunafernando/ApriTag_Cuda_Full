"""
Phase 14B smoke test: CPU vs GPU Canny edge detection accuracy (5 frames).
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
    frames = load_frames(video_path, range(5))

    # Default parameters (matching phase14B.md defaults)
    low_thresh = 35
    high_thresh = 110
    aperture_size = 3
    use_l2_gradient = True

    csv_path = Path("outputs/phase14B_edges_smoke5.csv")
    report_path = Path("outputs/phase14B_edges_smoke5_report.txt")

    rows = []
    all_passed = True

    for idx, frame in enumerate(frames):
        # Convert to grayscale
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # CPU path
        edges_cpu = cpu_canny_edges(
            gray,
            low_thresh=low_thresh,
            high_thresh=high_thresh,
            aperture_size=aperture_size,
            use_l2_gradient=use_l2_gradient,
        )

        # GPU path
        gray_gpu = cp.asarray(gray, dtype=cp.uint8)
        edges_gpu = gpu_canny_edges(
            gray_gpu,
            low_thresh=float(low_thresh),
            high_thresh=float(high_thresh),
            aperture_size=aperture_size,
            use_l2_gradient=use_l2_gradient,
        )
        edges_gpu_cpu = cp.asnumpy(edges_gpu)

        # Compute metrics
        edge_count_cpu = int(np.count_nonzero(edges_cpu == 255))
        edge_count_gpu = int(np.count_nonzero(edges_gpu_cpu == 255))
        edge_count_ratio = float(edge_count_gpu / edge_count_cpu) if edge_count_cpu > 0 else 0.0

        match_mask = (edges_cpu == edges_gpu_cpu)
        match_ratio = float(np.count_nonzero(match_mask)) / edges_cpu.size

        cpu_only = (edges_cpu == 255) & (edges_gpu_cpu == 0)
        gpu_only = (edges_cpu == 0) & (edges_gpu_cpu == 255)
        cpu_only_ratio = float(np.count_nonzero(cpu_only)) / edges_cpu.size
        gpu_only_ratio = float(np.count_nonzero(gpu_only)) / edges_cpu.size

        # Pass criteria
        passed = (
            0.8 <= edge_count_ratio <= 1.2
            and match_ratio >= 0.85
            and cpu_only_ratio <= 0.10
            and gpu_only_ratio <= 0.10
        )

        if not passed:
            all_passed = False

        rows.append([
            idx,
            edge_count_cpu,
            edge_count_gpu,
            edge_count_ratio,
            match_ratio,
            cpu_only_ratio,
            gpu_only_ratio,
            passed,
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
            "passed",
        ])
        writer.writerows(rows)

    # Write report
    report_lines = [
        "Phase 14B Smoke Test: CPU vs GPU Canny Edge Detection (5 frames)",
        "=" * 70,
        "",
        f"frames_processed: {len(frames)}",
        f"low_thresh: {low_thresh}",
        f"high_thresh: {high_thresh}",
        f"aperture_size: {aperture_size}",
        f"use_l2_gradient: {use_l2_gradient}",
        "",
        "Per-frame results:",
    ]

    for row in rows:
        idx, ec_cpu, ec_gpu, ec_ratio, match_r, cpu_only_r, gpu_only_r, passed = row
        status = "PASS" if passed else "FAIL"
        report_lines.append(
            f"  Frame {idx}: edge_count_ratio={ec_ratio:.4f}, match_ratio={match_r:.4f}, "
            f"cpu_only={cpu_only_r:.4f}, gpu_only={gpu_only_r:.4f} [{status}]"
        )

    report_lines.extend([
        "",
        f"Overall: {'PASS' if all_passed else 'FAIL'}",
        "",
        "Pass criteria:",
        "  - 0.8 <= edge_count_ratio <= 1.2",
        "  - match_ratio >= 0.85",
        "  - cpu_only_ratio <= 0.10",
        "  - gpu_only_ratio <= 0.10",
    ])

    report_path.write_text("\n".join(report_lines))
    print(f"Phase 14B smoke test complete: {csv_path}")
    print(f"Report: {report_path}")
    print(f"Overall: {'PASS' if all_passed else 'FAIL'}")


if __name__ == "__main__":
    main()

