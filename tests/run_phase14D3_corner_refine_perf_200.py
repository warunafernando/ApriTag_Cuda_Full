"""
Phase 14D_3: Compare CPU vs GPU vs AUTO corner refinement performance (200 frames).
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

    video_path = cfg["input"]["video_path"]
    frames = load_frames(video_path, range(200))

    edges_cfg = cfg.get("edges", {})
    low_thresh = int(edges_cfg.get("low_thresh", 35))
    high_thresh = int(edges_cfg.get("high_thresh", 110))

    modes = ["CPU", "GPU", "AUTO"]
    results = {mode: [] for mode in modes}

    for idx, frame in enumerate(frames):
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges_cpu = cpu_canny_edges(gray, low_thresh=low_thresh, high_thresh=high_thresh)
        quads_cpu = cpu_quad_candidates_from_edges(edges_cpu, cfg)
        quad_best = select_best_quad(quads_cpu)

        if quad_best is None:
            for mode in modes:
                results[mode].append([idx, 0.0])
            continue

        corners_in = quad_best.reshape(1, 4, 2).astype(np.float32)
        image_shape = (gray.shape[0], gray.shape[1])

        for mode in modes:
            # Update config for this mode
            cfg_copy = cfg.copy()
            cfg_copy["corner_refine"] = cfg["corner_refine"].copy()
            cfg_copy["corner_refine"]["mode"] = mode

            t0 = time.perf_counter()
            _, timings = dispatch_corner_refine(
                gray, corners_in, cfg_copy, image_shape, num_tags=1
            )
            t_elapsed = (time.perf_counter() - t0) * 1000.0

            # Use timing from dispatch if available and > 0, otherwise use elapsed time
            refine_ms = timings.get("t_refine_ms", 0.0)
            if refine_ms <= 0:
                refine_ms = t_elapsed
            results[mode].append([idx, refine_ms])

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/200 frames...")

    # Write CSVs and compute summaries
    summary_lines = [
        "Phase 14D_3 Corner Refinement Performance Comparison (200 frames)",
        "=" * 70,
        "",
    ]

    for mode in modes:
        csv_path = Path(f"outputs/phase14D3_perf_200_{mode.lower()}.csv")
        write_csv(
            csv_path,
            ["frame_index", "t_refine_ms"],
            results[mode],
        )

        # Compute summary stats
        t_list = [r[1] for r in results[mode] if r[1] > 0]
        mean_refine = np.mean(t_list) if len(t_list) > 0 else 0.0
        median_refine = np.median(t_list) if len(t_list) > 0 else 0.0
        p90_refine = np.percentile(t_list, 90) if len(t_list) > 0 else 0.0
        p99_refine = np.percentile(t_list, 99) if len(t_list) > 0 else 0.0
        max_refine = np.max(t_list) if len(t_list) > 0 else 0.0

        summary_lines.extend([
            f"{mode} Mode:",
            f"  mean_refine_ms: {mean_refine:.4f}",
            f"  median_refine_ms: {median_refine:.4f}",
            f"  p90_refine_ms: {p90_refine:.4f}",
            f"  p99_refine_ms: {p99_refine:.4f}",
            f"  max_refine_ms: {max_refine:.4f}",
            "",
        ])

    # Decision recommendation
    cpu_mean = np.mean([r[1] for r in results["CPU"] if r[1] > 0])
    gpu_mean = np.mean([r[1] for r in results["GPU"] if r[1] > 0])
    auto_mean = np.mean([r[1] for r in results["AUTO"] if r[1] > 0])

    summary_lines.extend([
        "Decision Recommendation:",
        f"  CPU mean: {cpu_mean:.4f} ms",
        f"  GPU mean: {gpu_mean:.4f} ms",
        f"  AUTO mean: {auto_mean:.4f} ms",
        "",
    ])

    if cpu_mean < gpu_mean:
        summary_lines.append("  Recommendation: Use CPU mode (faster)")
    elif gpu_mean < cpu_mean * 0.5:
        summary_lines.append("  Recommendation: Use GPU mode (significantly faster)")
    else:
        summary_lines.append("  Recommendation: Use AUTO mode (adaptive)")

    # Write summary report
    report_path = Path("outputs/phase14D3_perf_200_report.txt")
    report_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Phase 14D_3 performance comparison complete: {report_path}")


if __name__ == "__main__":
    main()

