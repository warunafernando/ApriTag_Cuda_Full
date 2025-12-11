"""
Phase 14C smoke test: CPU vs GPU quad candidate extraction from edges (5 frames).
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
from cpu.quads import cpu_quad_candidates_from_edges
from tests.helpers import rms

try:
    import cupy as cp
    from gpu.quads import gpu_quad_candidates_from_edges
except Exception as exc:
    cp = None
    gpu_quad_candidates_from_edges = None
    _gpu_import_error = exc
else:
    _gpu_import_error = None


def select_best_quad(quads: np.ndarray, expected_id: int | None = None) -> np.ndarray | None:
    """
    Select best quad candidate based on area and center position.
    For now, just pick the largest area quad.
    """
    if quads.shape[0] == 0:
        return None

    # Compute areas
    areas = []
    for quad in quads:
        area = cv2.contourArea(quad)
        areas.append(area)

    # Pick largest
    best_idx = np.argmax(areas)
    return quads[best_idx]


def main() -> None:
    cfg = load_config()
    ensure_output_dirs(cfg)

    if gpu_quad_candidates_from_edges is None:
        raise RuntimeError(f"GPU quads unavailable: {_gpu_import_error}")

    video_path = cfg["input"]["video_path"]
    frames = load_frames(video_path, range(5))

    # Get edge parameters
    edges_cfg = cfg.get("edges", {})
    low_thresh = int(edges_cfg.get("low_thresh", 35))
    high_thresh = int(edges_cfg.get("high_thresh", 110))

    csv_path = Path("outputs/phase14C_quads_smoke5.csv")
    report_path = Path("outputs/phase14C_quads_smoke5_report.txt")
    debug_dir = Path("outputs/debug_phase14C")
    debug_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    all_passed = True

    for idx, frame in enumerate(frames):
        # Convert to grayscale
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Generate edges (CPU Canny)
        edges_cpu = cpu_canny_edges(
            gray,
            low_thresh=low_thresh,
            high_thresh=high_thresh,
        )

        # CPU quads
        quads_cpu = cpu_quad_candidates_from_edges(edges_cpu, cfg)
        num_quads_cpu = quads_cpu.shape[0]

        # GPU quads
        edges_gpu = cp.asarray(edges_cpu, dtype=cp.uint8)
        quads_gpu, timings = gpu_quad_candidates_from_edges(edges_gpu, cfg)
        quads_gpu_cpu = cp.asnumpy(quads_gpu) if quads_gpu.shape[0] > 0 else np.empty((0, 4, 2), dtype=np.float32)
        num_quads_gpu = quads_gpu_cpu.shape[0]

        # Select best quad
        quad_cpu_best = select_best_quad(quads_cpu)
        quad_gpu_best = select_best_quad(quads_gpu_cpu)

        # Compute RMS
        if quad_cpu_best is not None and quad_gpu_best is not None:
            quad_rms_px = rms(quad_cpu_best, quad_gpu_best)
        else:
            quad_rms_px = float('inf')

        # Pass criteria
        passed = (
            num_quads_cpu >= 1
            and num_quads_gpu >= 1
            and quad_rms_px <= 2.0
        )

        if not passed:
            all_passed = False

        rows.append([
            idx,
            num_quads_cpu,
            num_quads_gpu,
            quad_rms_px,
        ])

        # Save overlay
        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else frame.copy()

        # Draw CPU quads (blue)
        for quad in quads_cpu:
            pts = quad.reshape(-1, 1, 2).astype(int)
            cv2.polylines(overlay, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

        # Draw GPU quads (green)
        for quad in quads_gpu_cpu:
            pts = quad.reshape(-1, 1, 2).astype(int)
            cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # Draw best quads (thicker)
        if quad_cpu_best is not None:
            pts = quad_cpu_best.reshape(-1, 1, 2).astype(int)
            cv2.polylines(overlay, [pts], isClosed=True, color=(255, 0, 0), thickness=3)
        if quad_gpu_best is not None:
            pts = quad_gpu_best.reshape(-1, 1, 2).astype(int)
            cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 0), thickness=3)

        cv2.imwrite(str(debug_dir / f"frame_{idx:04d}_quads.png"), overlay)

    # Write CSV
    from tests.helpers import write_csv
    write_csv(
        csv_path,
        ["frame_index", "num_quads_cpu", "num_quads_gpu", "quad_rms_px"],
        rows,
    )

    # Write report
    report_lines = [
        "Phase 14C Smoke Test: CPU vs GPU Quad Extraction (5 frames)",
        "=" * 70,
        "",
        f"frames_processed: {len(frames)}",
        "",
        "Per-frame results:",
    ]

    for row in rows:
        idx, n_cpu, n_gpu, rms_px = row
        passed = n_cpu >= 1 and n_gpu >= 1 and rms_px <= 2.0
        status = "PASS" if passed else "FAIL"
        report_lines.append(
            f"  Frame {idx}: num_quads_cpu={n_cpu}, num_quads_gpu={n_gpu}, "
            f"quad_rms_px={rms_px:.4f} [{status}]"
        )

    report_lines.extend([
        "",
        f"Overall: {'PASS' if all_passed else 'FAIL'}",
        "",
        "Pass criteria:",
        "  - num_quads_cpu >= 1",
        "  - num_quads_gpu >= 1",
        "  - quad_rms_px <= 2.0",
    ])

    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Phase 14C smoke test complete: {csv_path}")
    print(f"Report: {report_path}")
    print(f"Overall: {'PASS' if all_passed else 'FAIL'}")


if __name__ == "__main__":
    main()

