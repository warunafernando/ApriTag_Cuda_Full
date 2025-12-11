"""
Phase 14D_2 diagnostic tool: Per-corner error analysis for GPU corner refinement.
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
from cpu.corner_refine import cpu_refine_corners
from cpu.edges import cpu_canny_edges
from cpu.quads import cpu_quad_candidates_from_edges
from gpu.corner_refine import gpu_refine_corners

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

    if cp is None:
        raise RuntimeError(f"CuPy not available: {_gpu_import_error}")

    video_path = cfg["input"]["video_path"]
    frames = load_frames(video_path, range(5))

    # Get edge parameters
    edges_cfg = cfg.get("edges", {})
    low_thresh = int(edges_cfg.get("low_thresh", 35))
    high_thresh = int(edges_cfg.get("high_thresh", 110))

    # Get corner refinement parameters
    refine_cfg = cfg.get("corner_refine", {})
    window_size = int(refine_cfg.get("window_size", 5))
    max_iters = int(refine_cfg.get("max_iters", 10))
    epsilon = float(refine_cfg.get("epsilon", 0.01))

    output_path = Path("outputs/phase14D2_corner_refine_smoke5_debug.txt")

    lines = [
        "Phase 14D_2 Diagnostic: Per-Corner Error Analysis",
        "=" * 70,
        "",
        f"window_size: {window_size}",
        f"max_iters: {max_iters}",
        f"epsilon: {epsilon}",
        "",
    ]

    for idx, frame in enumerate(frames):
        # Convert to grayscale
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Generate edges (CPU Canny)
        edges_cpu = cpu_canny_edges(gray, low_thresh=low_thresh, high_thresh=high_thresh)

        # Get quads (using CPU quads for both paths to ensure same input)
        quads_cpu = cpu_quad_candidates_from_edges(edges_cpu, cfg)
        quad_best = select_best_quad(quads_cpu)

        if quad_best is None:
            lines.append(f"frame {idx}: no quads found")
            continue

        # Prepare corners for refinement (need (1, 4, 2) shape)
        corners_in = quad_best.reshape(1, 4, 2).astype(np.float32)

        # CPU refinement
        corners_cpu = cpu_refine_corners(gray, corners_in, window_size, max_iters, epsilon)
        corners_cpu = corners_cpu[0]  # Get first quad

        # GPU refinement
        gray_gpu = cp.asarray(gray, dtype=cp.uint8)
        corners_in_gpu = cp.asarray(corners_in, dtype=cp.float32)
        corners_gpu, timings = gpu_refine_corners(gray_gpu, corners_in_gpu, window_size, max_iters, epsilon)
        corners_gpu_cpu = cp.asnumpy(corners_gpu[0])  # Get first quad

        lines.append(f"frame {idx}")
        for corner_idx in range(4):
            x_cpu = corners_cpu[corner_idx, 0]
            y_cpu = corners_cpu[corner_idx, 1]
            x_gpu = corners_gpu_cpu[corner_idx, 0]
            y_gpu = corners_gpu_cpu[corner_idx, 1]

            dx = x_gpu - x_cpu
            dy = y_gpu - y_cpu
            dist = np.sqrt(dx**2 + dy**2)

            lines.append(
                f"  quad 0 corner {corner_idx}: "
                f"dx={dx:.4f}, dy={dy:.4f}, dist={dist:.4f}, "
                f"cpu=({x_cpu:.2f}, {y_cpu:.2f}), gpu=({x_gpu:.2f}, {y_gpu:.2f})"
            )

        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Phase 14D_2 diagnostic complete: {output_path}")


if __name__ == "__main__":
    main()

