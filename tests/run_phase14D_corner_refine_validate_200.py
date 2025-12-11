"""
Phase 14D validation: CPU vs GPU corner refinement in full pipeline (200 frames).
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
from common.geometry import canonicalize_corners
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
    frames = load_frames(video_path, range(200))

    # Get edge parameters
    edges_cfg = cfg.get("edges", {})
    low_thresh = int(edges_cfg.get("low_thresh", 35))
    high_thresh = int(edges_cfg.get("high_thresh", 110))
    expected_id = cfg.get("tag", {}).get("id_expected", 3)

    # Get corner refinement parameters
    refine_cfg = cfg.get("corner_refine", {})
    window_size = int(refine_cfg.get("window_size", 5))
    max_iters = int(refine_cfg.get("max_iters", 10))
    epsilon = float(refine_cfg.get("epsilon", 0.01))

    csv_path = Path("outputs/phase14D_corner_refine_200.csv")
    report_path = Path("outputs/phase14D_corner_refine_200_report.txt")

    rows = []

    for idx, frame in enumerate(frames):
        # Convert to grayscale
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Generate edges (CPU Canny)
        edges_cpu = cpu_canny_edges(gray, low_thresh=low_thresh, high_thresh=high_thresh)

        # Get quads (using CPU quads for both paths to ensure same input)
        quads_cpu = cpu_quad_candidates_from_edges(edges_cpu, cfg)
        quad_best = select_best_quad(quads_cpu)

        if quad_best is None:
            rows.append([
                idx, None, None, None, None, False, False,
                float('inf'), float('inf'), float('inf'), 0.0, 0.0,
            ])
            continue

        # Prepare corners for refinement (need (1, 4, 2) shape)
        corners_in = quad_best.reshape(1, 4, 2).astype(np.float32)

        # CPU refinement with timing
        t0 = time.perf_counter()
        corners_cpu = cpu_refine_corners(gray, corners_in, window_size, max_iters, epsilon)
        t_cpu_refine = (time.perf_counter() - t0) * 1000.0
        corners_cpu = corners_cpu[0]  # Get first quad

        # GPU refinement with timing
        gray_gpu = cp.asarray(gray, dtype=cp.uint8)
        corners_in_gpu = cp.asarray(corners_in, dtype=cp.float32)
        t0 = time.perf_counter()
        corners_gpu, timings = gpu_refine_corners(gray_gpu, corners_in_gpu, window_size, max_iters, epsilon)
        t_gpu_refine = timings["t_refine_ms"]
        corners_gpu_cpu = cp.asnumpy(corners_gpu[0])  # Get first quad

        # Run decode for both
        decode_cpu = decode_apriltag(gray, corners_cpu, cfg)
        id_cpu = decode_cpu.id
        hamming_cpu = decode_cpu.hamming

        sample_grid_gpu = sample_gpu(gray, corners_gpu_cpu, cfg)
        sample_grid_cpu = sample_grid_gpu if isinstance(sample_grid_gpu, np.ndarray) else cp.asnumpy(sample_grid_gpu)
        gpu_bits = decode_gpu_bits(sample_grid_cpu, cfg)
        id_gpu, hamming_gpu = decode_gpu_codebook(gpu_bits, cfg)

        # Compute metrics
        id_match = (id_cpu == id_gpu == expected_id) if (id_cpu is not None and id_gpu is not None) else False
        hamming_match = (hamming_cpu == hamming_gpu == 0) if (hamming_cpu is not None and hamming_gpu is not None) else False

        # Corner RMS
        corner_rms_px = rms(corners_cpu, corners_gpu_cpu)

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

        rows.append([
            idx,
            id_cpu,
            id_gpu,
            hamming_cpu if hamming_cpu is not None else -1,
            hamming_gpu if hamming_gpu is not None else -1,
            id_match,
            hamming_match,
            corner_rms_px,
            rot_err_deg,
            trans_err_m,
            t_cpu_refine,
            t_gpu_refine,
        ])

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/200 frames...")

    # Write CSV
    write_csv(
        csv_path,
        [
            "frame_index",
            "id_cpu",
            "id_gpu",
            "hamming_cpu",
            "hamming_gpu",
            "id_match",
            "hamming_match",
            "corner_rms_px",
            "rot_err_deg",
            "trans_err_m",
            "t_cpu_refine_ms",
            "t_gpu_refine_ms",
        ],
        rows,
    )

    # Compute summary statistics
    id_matches = [r[5] for r in rows if r[5] is not None]
    hamming_matches = [r[6] for r in rows if r[6] is not None]
    corner_rms_list = [r[7] for r in rows if r[7] != float('inf')]
    rot_err_list = [r[8] for r in rows if r[8] != float('inf')]
    trans_err_list = [r[9] for r in rows if r[9] != float('inf')]
    t_cpu_list = [r[10] for r in rows if r[10] > 0]
    t_gpu_list = [r[11] for r in rows if r[11] > 0]

    id_match_rate = sum(id_matches) / len(id_matches) if len(id_matches) > 0 else 0.0
    hamming_match_rate = sum(hamming_matches) / len(hamming_matches) if len(hamming_matches) > 0 else 0.0
    mean_corner_rms = np.mean(corner_rms_list) if len(corner_rms_list) > 0 else float('inf')
    max_corner_rms = np.max(corner_rms_list) if len(corner_rms_list) > 0 else float('inf')
    mean_rot_err = np.mean(rot_err_list) if len(rot_err_list) > 0 else float('inf')
    max_rot_err = np.max(rot_err_list) if len(rot_err_list) > 0 else float('inf')
    mean_trans_err = np.mean(trans_err_list) if len(trans_err_list) > 0 else float('inf')
    max_trans_err = np.max(trans_err_list) if len(trans_err_list) > 0 else float('inf')
    mean_cpu_refine = np.mean(t_cpu_list) if len(t_cpu_list) > 0 else 0.0
    mean_gpu_refine = np.mean(t_gpu_list) if len(t_gpu_list) > 0 else 0.0
    refine_speedup = mean_cpu_refine / mean_gpu_refine if mean_gpu_refine > 0 else 0.0

    # Write report
    report_lines = [
        "Phase 14D Validation: CPU vs GPU Corner Refinement (200 frames)",
        "=" * 70,
        "",
        f"frames_processed: {len(frames)}",
        "",
        "Accuracy Metrics:",
        f"  id_match_rate: {id_match_rate:.4f}",
        f"  hamming_match_rate: {hamming_match_rate:.4f}",
        f"  mean_corner_rms_px: {mean_corner_rms:.4f}",
        f"  max_corner_rms_px: {max_corner_rms:.4f}",
        f"  mean_rot_err_deg: {mean_rot_err:.4f}",
        f"  max_rot_err_deg: {max_rot_err:.4f}",
        f"  mean_trans_err_m: {mean_trans_err:.4f}",
        f"  max_trans_err_m: {max_trans_err:.4f}",
        "",
        "Performance Metrics:",
        f"  mean_cpu_refine_ms: {mean_cpu_refine:.4f}",
        f"  mean_gpu_refine_ms: {mean_gpu_refine:.4f}",
        f"  refine_speedup: {refine_speedup:.4f}x",
        "",
        "Pass Criteria:",
        "  - id_match_rate >= 0.99",
        "  - hamming_match_rate >= 0.99",
        "  - mean_corner_rms_px <= 0.25",
        "  - max_corner_rms_px <= 0.5",
        "  - mean_rot_err_deg <= 0.3",
        "  - max_rot_err_deg <= 0.8",
        "  - mean_trans_err_m <= 0.005",
        "  - max_trans_err_m <= 0.015",
        "  - mean_gpu_refine_ms < mean_cpu_refine_ms (desired: <= 0.4 ms)",
        "",
    ]

    # Check pass/fail
    passed = (
        id_match_rate >= 0.99
        and hamming_match_rate >= 0.99
        and mean_corner_rms <= 0.25
        and max_corner_rms <= 0.5
        and mean_rot_err <= 0.3
        and max_rot_err <= 0.8
        and mean_trans_err <= 0.005
        and max_trans_err <= 0.015
        and mean_gpu_refine < mean_cpu_refine
    )

    report_lines.append(f"Overall: {'PASS' if passed else 'FAIL'}")

    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Phase 14D validation complete: {csv_path}")
    print(f"Report: {report_path}")
    print(f"Overall: {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()

