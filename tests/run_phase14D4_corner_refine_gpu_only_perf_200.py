"""
Phase 14D_4: GPU-only corner refinement performance test (200 frames).
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
from common.corner_refine_dispatch import dispatch_corner_refine
from common.video import load_frames
from cpu.edges import cpu_canny_edges
from cpu.quads import cpu_quad_candidates_from_edges
from tests.helpers import write_csv

try:
    import cupy as cp
except Exception as exc:
    cp = None
    _gpu_import_error = exc
else:
    _gpu_import_error = None


def select_best_quad(quads: np.ndarray) -> np.ndarray | None:
    """Select best quad candidate based on area."""
    if quads.shape[0] == 0:
        return None
    areas = [cv2.contourArea(quad) for quad in quads]
    best_idx = np.argmax(areas)
    return quads[best_idx]


def main() -> None:
    cfg = load_config()
    ensure_output_dirs(cfg)

    # Force GPU mode
    cfg["corner_refine"]["mode"] = "GPU"
    cfg["corner_refine"]["allow_failover"] = False

    video_path = cfg["input"]["video_path"]
    frames = load_frames(video_path, range(200))

    edges_cfg = cfg.get("edges", {})
    low_thresh = int(edges_cfg.get("low_thresh", 35))
    high_thresh = int(edges_cfg.get("high_thresh", 110))

    csv_path = Path("outputs/phase14D4_corner_refine_gpu_perf_200.csv")
    report_path = Path("outputs/phase14D4_corner_refine_gpu_perf_200_report.txt")

    rows = []

    for idx, frame in enumerate(frames):
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges_cpu = cpu_canny_edges(gray, low_thresh=low_thresh, high_thresh=high_thresh)
        quads_cpu = cpu_quad_candidates_from_edges(edges_cpu, cfg)
        quad_best = select_best_quad(quads_cpu)

        if quad_best is None:
            rows.append([idx, 0.0])
            continue

        corners_in = quad_best.reshape(1, 4, 2).astype(np.float32)
        image_shape = (gray.shape[0], gray.shape[1])

        _, timings = dispatch_corner_refine(gray, corners_in, cfg, image_shape, num_tags=1)
        refine_ms = timings.get("t_refine_ms", 0.0)
        rows.append([idx, refine_ms])

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/200 frames...")

    # Write CSV
    write_csv(
        csv_path,
        ["frame_index", "t_gpu_refine_ms"],
        rows,
    )

    # Compute summary
    t_list = [r[1] for r in rows if r[1] > 0]
    mean_refine = np.mean(t_list) if len(t_list) > 0 else 0.0
    median_refine = np.median(t_list) if len(t_list) > 0 else 0.0
    p90_refine = np.percentile(t_list, 90) if len(t_list) > 0 else 0.0
    p99_refine = np.percentile(t_list, 99) if len(t_list) > 0 else 0.0
    max_refine = np.max(t_list) if len(t_list) > 0 else 0.0

    perf_passed = mean_refine <= 5.0

    # Write report
    report_lines = [
        "Phase 14D_4 GPU-Only Corner Refinement Performance (200 frames)",
        "=" * 70,
        "",
        f"frames_processed: {len(frames)}",
        "",
        "Performance Metrics:",
        f"  mean_gpu_refine_ms: {mean_refine:.4f} {'✅' if perf_passed else '❌'} (target: ≤ 5.0)",
        f"  median_gpu_refine_ms: {median_refine:.4f}",
        f"  p90_gpu_refine_ms: {p90_refine:.4f}",
        f"  p99_gpu_refine_ms: {p99_refine:.4f}",
        f"  max_gpu_refine_ms: {max_refine:.4f}",
        "",
        f"Performance Target: {'PASS' if perf_passed else 'FAIL'}",
        "",
        "Note: Stretch goal is ≤ 2.0 ms",
    ]

    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Phase 14D_4 GPU-only performance test complete: {csv_path}")
    print(f"Report: {report_path}")
    print(f"Performance: {'PASS' if perf_passed else 'FAIL'}")


if __name__ == "__main__":
    main()

