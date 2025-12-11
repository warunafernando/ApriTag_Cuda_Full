"""
Phase 14D smoke test: CPU vs GPU corner refinement (5 frames).
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
from common.pose import rotation_error_deg, solve_pose_cpu, solve_pose_gpu_from_homography
from common.video import load_frames
from cpu.corner_refine import cpu_refine_corners
from cpu.decode import decode_apriltag
from cpu.edges import cpu_canny_edges
from cpu.quads import cpu_quad_candidates_from_edges
from gpu.corner_refine import gpu_refine_corners
from gpu.decode import decode_gpu_bits, decode_gpu_codebook
from gpu.quads import gpu_quad_candidates_from_edges
from gpu.sampling import sample_gpu
from tests.helpers import rms, write_csv

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

    csv_path = Path("outputs/phase14D_corner_refine_smoke5.csv")
    report_path = Path("outputs/phase14D_corner_refine_smoke5_report.txt")
    debug_dir = Path("outputs/debug_phase14D")
    debug_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    all_passed = True

    for idx, frame in enumerate(frames):
        # Convert to grayscale
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Generate edges (CPU Canny)
        edges_cpu = cpu_canny_edges(gray, low_thresh=low_thresh, high_thresh=high_thresh)

        # Get quads (using CPU quads for both paths to ensure same input)
        quads_cpu = cpu_quad_candidates_from_edges(edges_cpu, cfg)
        quad_best = select_best_quad(quads_cpu)

        if quad_best is None:
            rows.append([idx, float('inf'), float('inf'), float('inf')])
            all_passed = False
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

        # Compute corner RMS
        corner_rms_px = rms(corners_cpu, corners_gpu_cpu)

        # Run decode and pose for both
        decode_cpu = decode_apriltag(gray, corners_cpu, cfg)
        sample_grid_gpu = sample_gpu(gray, corners_gpu_cpu, cfg)
        sample_grid_cpu = sample_grid_gpu if isinstance(sample_grid_gpu, np.ndarray) else cp.asnumpy(sample_grid_gpu)
        gpu_bits = decode_gpu_bits(sample_grid_cpu, cfg)
        id_gpu, hamming_gpu = decode_gpu_codebook(gpu_bits, cfg)

        # Pose
        image_shape = (gray.shape[0], gray.shape[1])
        pose_cpu = solve_pose_cpu(corners_cpu, image_shape, cfg)
        pose_gpu = solve_pose_gpu_from_homography(corners_gpu_cpu, image_shape, cfg)

        # Compute pose errors
        if pose_cpu is not None and pose_gpu is not None:
            rvec_cpu, tvec_cpu = pose_cpu
            rvec_gpu, tvec_gpu = pose_gpu
            R_cpu, _ = cv2.Rodrigues(rvec_cpu)
            R_gpu, _ = cv2.Rodrigues(rvec_gpu)
            rot_err_deg = rotation_error_deg(R_cpu, R_gpu)
            trans_err_m = float(np.linalg.norm(tvec_cpu - tvec_gpu))
        else:
            rot_err_deg = float('inf')
            trans_err_m = float('inf')

        # Pass criteria
        passed = (
            corner_rms_px <= 0.3
            and rot_err_deg <= 0.3
            and trans_err_m <= 0.005
        )

        if not passed:
            all_passed = False

        rows.append([idx, corner_rms_px, rot_err_deg, trans_err_m])

        # Save overlay
        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else frame.copy()

        # Draw CPU corners (blue)
        pts_cpu = corners_cpu.reshape(-1, 1, 2).astype(int)
        cv2.polylines(overlay, [pts_cpu], isClosed=True, color=(255, 0, 0), thickness=2)
        for i, pt in enumerate(corners_cpu):
            cv2.circle(overlay, tuple(pt.astype(int)), 3, (255, 0, 0), -1)

        # Draw GPU corners (green)
        pts_gpu = corners_gpu_cpu.reshape(-1, 1, 2).astype(int)
        cv2.polylines(overlay, [pts_gpu], isClosed=True, color=(0, 255, 0), thickness=2)
        for i, pt in enumerate(corners_gpu_cpu):
            cv2.circle(overlay, tuple(pt.astype(int)), 3, (0, 255, 0), -1)

        cv2.imwrite(str(debug_dir / f"frame_{idx:04d}_refine.png"), overlay)

    # Write CSV
    write_csv(
        csv_path,
        ["frame_index", "corner_rms_px", "rot_err_deg", "trans_err_m"],
        rows,
    )

    # Write report
    report_lines = [
        "Phase 14D Smoke Test: CPU vs GPU Corner Refinement (5 frames)",
        "=" * 70,
        "",
        f"frames_processed: {len(frames)}",
        f"window_size: {window_size}",
        f"max_iters: {max_iters}",
        f"epsilon: {epsilon}",
        "",
        "Per-frame results:",
    ]

    for row in rows:
        idx, rms_px, rot_err, trans_err = row
        passed = rms_px <= 0.3 and rot_err <= 0.3 and trans_err <= 0.005
        status = "PASS" if passed else "FAIL"
        report_lines.append(
            f"  Frame {idx}: corner_rms_px={rms_px:.4f}, "
            f"rot_err_deg={rot_err:.4f}, trans_err_m={trans_err:.4f} [{status}]"
        )

    report_lines.extend([
        "",
        f"Overall: {'PASS' if all_passed else 'FAIL'}",
        "",
        "Pass criteria:",
        "  - corner_rms_px <= 0.3",
        "  - rot_err_deg <= 0.3",
        "  - trans_err_m <= 0.005",
    ])

    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Phase 14D smoke test complete: {csv_path}")
    print(f"Report: {report_path}")
    print(f"Overall: {'PASS' if all_passed else 'FAIL'}")


if __name__ == "__main__":
    main()

