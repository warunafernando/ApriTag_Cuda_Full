"""
Phase 14B_3 smoke test: Fast GPU edges (5 frames).
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
    from gpu.edges_fast import gpu_fast_edges
except Exception as exc:
    cp = None
    gpu_fast_edges = None
    _gpu_import_error = exc
else:
    _gpu_import_error = None


def main() -> None:
    cfg = load_config()
    ensure_output_dirs(cfg)

    if gpu_fast_edges is None:
        raise RuntimeError(f"GPU fast edges unavailable: {_gpu_import_error}")

    video_path = cfg["input"]["video_path"]
    frames = load_frames(video_path, range(5))

    # Parameters
    low_thresh = 30.0
    high_thresh = 90.0
    # CPU Canny params for reference
    cpu_low = 35
    cpu_high = 110

    csv_path = Path("outputs/phase14B3_fast_edges_smoke5.csv")
    report_path = Path("outputs/phase14B3_fast_edges_smoke5_report.txt")

    rows = []
    all_passed = True

    for idx, frame in enumerate(frames):
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # CPU Canny (for reference)
        t0 = time.perf_counter()
        edges_cpu = cpu_canny_edges(gray, low_thresh=cpu_low, high_thresh=cpu_high)
        t_cpu = (time.perf_counter() - t0) * 1000.0

        # GPU fast edges
        gray_gpu = cp.asarray(gray, dtype=cp.uint8)
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        edges_gpu = gpu_fast_edges(gray_gpu, low_thresh=low_thresh, high_thresh=high_thresh)
        cp.cuda.Stream.null.synchronize()
        t_gpu = (time.perf_counter() - t0) * 1000.0

        edges_gpu_cpu = cp.asnumpy(edges_gpu)

        # Metrics
        edge_count_cpu = int(np.count_nonzero(edges_cpu == 255))
        edge_count_fast_gpu = int(np.count_nonzero(edges_gpu_cpu == 255))
        edge_count_ratio = float(edge_count_fast_gpu / edge_count_cpu) if edge_count_cpu > 0 else 0.0

        match_mask = (edges_cpu == edges_gpu_cpu)
        match_ratio = float(np.count_nonzero(match_mask)) / edges_cpu.size

        # Pass criteria (looser than Canny)
        passed = (
            0.4 <= edge_count_ratio <= 1.5
            and match_ratio >= 0.70
            and t_gpu < 4.0 * t_cpu
        )

        if not passed:
            all_passed = False

        rows.append([
            idx,
            edge_count_cpu,
            edge_count_fast_gpu,
            edge_count_ratio,
            match_ratio,
            t_cpu,
            t_gpu,
            passed,
        ])

    # Write CSV
    import csv
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_index",
            "edge_count_cpu",
            "edge_count_fast_gpu",
            "edge_count_ratio",
            "match_ratio",
            "t_cpu_edges_ms",
            "t_gpu_fast_edges_ms",
            "passed",
        ])
        writer.writerows(rows)

    # Write report
    report_lines = [
        "Phase 14B_3 Smoke Test: Fast GPU Edges (5 frames)",
        "=" * 70,
        "",
        f"frames_processed: {len(frames)}",
        f"low_thresh: {low_thresh}",
        f"high_thresh: {high_thresh}",
        "",
        "Per-frame results:",
    ]

    for row in rows:
        idx, ec_cpu, ec_gpu, ec_ratio, match_r, t_cpu, t_gpu, passed = row
        status = "PASS" if passed else "FAIL"
        report_lines.append(
            f"  Frame {idx}: edge_ratio={ec_ratio:.4f}, match={match_r:.4f}, "
            f"t_cpu={t_cpu:.3f}ms, t_gpu={t_gpu:.3f}ms [{status}]"
        )

    report_lines.extend([
        "",
        f"Overall: {'PASS' if all_passed else 'FAIL'}",
        "",
        "Pass criteria:",
        "  - 0.4 <= edge_count_ratio <= 1.5",
        "  - match_ratio >= 0.70",
        "  - t_gpu_fast_edges_ms < 4.0 * t_cpu_edges_ms",
    ])

    report_path.write_text("\n".join(report_lines))
    print(f"Phase 14B_3 smoke test complete: {csv_path}")
    print(f"Report: {report_path}")
    print(f"Overall: {'PASS' if all_passed else 'FAIL'}")


if __name__ == "__main__":
    main()

