"""
Phase 14D_4: Block size sweep for GPU corner refinement performance tuning.
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
    frames = load_frames(video_path, range(50))  # Use 50 frames for sweep

    edges_cfg = cfg.get("edges", {})
    low_thresh = int(edges_cfg.get("low_thresh", 35))
    high_thresh = int(edges_cfg.get("high_thresh", 110))

    block_sizes = [32, 64, 128, 256]
    results = []

    for block_size in block_sizes:
        cfg_copy = cfg.copy()
        cfg_copy["corner_refine"] = cfg["corner_refine"].copy()
        cfg_copy["corner_refine"]["mode"] = "GPU"
        cfg_copy["corner_refine"]["allow_failover"] = False
        cfg_copy["corner_refine"]["threads_per_block"] = block_size

        timings_list = []

        for idx, frame in enumerate(frames):
            gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges_cpu = cpu_canny_edges(gray, low_thresh=low_thresh, high_thresh=high_thresh)
            quads_cpu = cpu_quad_candidates_from_edges(edges_cpu, cfg)
            quad_best = select_best_quad(quads_cpu)

            if quad_best is None:
                continue

            corners_in = quad_best.reshape(1, 4, 2).astype(np.float32)
            image_shape = (gray.shape[0], gray.shape[1])

            _, timings = dispatch_corner_refine(gray, corners_in, cfg_copy, image_shape, num_tags=1)
            refine_ms = timings.get("t_refine_ms", 0.0)
            if refine_ms > 0:
                timings_list.append(refine_ms)

        if len(timings_list) > 0:
            mean_ms = np.mean(timings_list)
            max_ms = np.max(timings_list)
            results.append((block_size, mean_ms, max_ms))
            print(f"Block size {block_size}: mean={mean_ms:.4f} ms, max={max_ms:.4f} ms")

    # Write results
    output_path = Path("outputs/phase14D4_block_sweep_gpu_refine.txt")
    lines = [
        "Phase 14D_4 Block Size Sweep for GPU Corner Refinement",
        "=" * 70,
        "",
        "Results (50 frames):",
        "",
    ]

    for block_size, mean_ms, max_ms in results:
        lines.append(f"  threads_per_block={block_size}: mean={mean_ms:.4f} ms, max={max_ms:.4f} ms")

    if len(results) > 0:
        best_block = min(results, key=lambda x: x[1])
        lines.extend([
            "",
            f"Best configuration: threads_per_block={best_block[0]} (mean={best_block[1]:.4f} ms)",
        ])

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Block size sweep complete: {output_path}")


if __name__ == "__main__":
    main()

