"""
Phase 14C validation: CPU vs GPU quads in full pipeline (200 frames).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import time

from common.config import ensure_output_dirs, load_config
from common.geometry import canonicalize_corners
from common.pose import solve_pose_cpu, solve_pose_gpu_from_homography
from common.video import load_frames
from cpu.decode import decode_apriltag
from cpu.edges import cpu_canny_edges
from cpu.quads import cpu_quad_candidates_from_edges
from gpu.decode import decode_gpu_bits, decode_gpu_codebook
from gpu.sampling import sample_gpu
from tests.helpers import rms, write_csv

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
    Select best quad candidate based on area.
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


def run_decode_with_quad(gray: np.ndarray, quad: np.ndarray, cfg: dict, use_gpu: bool) -> tuple[int | None, int | None]:
    """
    Run decode pipeline with given quad.
    Returns (id, hamming).
    """
    if use_gpu:
        # GPU decode path
        # sample_gpu expects numpy array when force_cpu_exact_sampling is True
        sample_grid_gpu = sample_gpu(gray, quad, cfg)
        # sample_grid_gpu might be cupy or numpy depending on force_cpu_exact_sampling
        if isinstance(sample_grid_gpu, cp.ndarray):
            sample_grid_cpu = cp.asnumpy(sample_grid_gpu)
        else:
            sample_grid_cpu = sample_grid_gpu
        gpu_bits = decode_gpu_bits(sample_grid_cpu, cfg)
        id_gpu, hamming_gpu = decode_gpu_codebook(gpu_bits, cfg)
        return id_gpu, hamming_gpu
    else:
        # CPU decode path
        decode_result = decode_apriltag(gray, quad, cfg)
        return decode_result.id, decode_result.hamming


def main() -> None:
    cfg = load_config()
    ensure_output_dirs(cfg)

    if gpu_quad_candidates_from_edges is None:
        raise RuntimeError(f"GPU quads unavailable: {_gpu_import_error}")

    video_path = cfg["input"]["video_path"]
    frames = load_frames(video_path, range(200))

    # Get edge parameters
    edges_cfg = cfg.get("edges", {})
    low_thresh = int(edges_cfg.get("low_thresh", 35))
    high_thresh = int(edges_cfg.get("high_thresh", 110))
    expected_id = cfg.get("tag", {}).get("id_expected", 3)

    csv_path = Path("outputs/phase14C_quads_200.csv")
    report_path = Path("outputs/phase14C_quads_200_report.txt")

    rows = []

    for idx, frame in enumerate(frames):
        # Convert to grayscale
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Generate edges (CPU Canny)
        edges_cpu = cpu_canny_edges(
            gray,
            low_thresh=low_thresh,
            high_thresh=high_thresh,
        )

        # Baseline: CPU quads
        t0 = time.perf_counter()
        quads_cpu = cpu_quad_candidates_from_edges(edges_cpu, cfg)
        t_cpu_quads = (time.perf_counter() - t0) * 1000.0
        num_quads_cpu = quads_cpu.shape[0]

        quad_cpu_best = select_best_quad(quads_cpu)
        if quad_cpu_best is not None:
            quad_cpu_best = canonicalize_corners(quad_cpu_best)
            id_cpu, hamming_cpu = run_decode_with_quad(gray, quad_cpu_best, cfg, use_gpu=False)
            corners_cpu = quad_cpu_best
            # Pose
            image_shape = (gray.shape[0], gray.shape[1])
            pose_cpu = solve_pose_cpu(corners_cpu, image_shape, cfg)
        else:
            id_cpu = None
            hamming_cpu = None
            corners_cpu = None
            pose_cpu = None

        # Under test: GPU quads
        edges_gpu = cp.asarray(edges_cpu, dtype=cp.uint8)
        t0 = time.perf_counter()
        quads_gpu, timings = gpu_quad_candidates_from_edges(edges_gpu, cfg)
        t_gpu_quads = timings["t_total_ms"]
        quads_gpu_cpu = cp.asnumpy(quads_gpu) if quads_gpu.shape[0] > 0 else np.empty((0, 4, 2), dtype=np.float32)
        num_quads_gpu = quads_gpu_cpu.shape[0]

        quad_gpu_best = select_best_quad(quads_gpu_cpu)
        if quad_gpu_best is not None:
            quad_gpu_best = canonicalize_corners(quad_gpu_best)
            id_gpu, hamming_gpu = run_decode_with_quad(gray, quad_gpu_best, cfg, use_gpu=True)
            corners_gpu = quad_gpu_best
            # Pose
            image_shape = (gray.shape[0], gray.shape[1])
            pose_gpu = solve_pose_gpu_from_homography(corners_gpu, image_shape, cfg)
        else:
            id_gpu = None
            hamming_gpu = None
            corners_gpu = None
            pose_gpu = None

        # Compute metrics
        id_match = (id_cpu == id_gpu == expected_id) if (id_cpu is not None and id_gpu is not None) else False
        hamming_match = (hamming_cpu == hamming_gpu == 0) if (hamming_cpu is not None and hamming_gpu is not None) else False

        if corners_cpu is not None and corners_gpu is not None:
            corner_rms_px = rms(corners_cpu, corners_gpu)
        else:
            corner_rms_px = float('inf')

        # Pose errors
        if pose_cpu is not None and pose_gpu is not None:
            rvec_cpu, tvec_cpu = pose_cpu
            rvec_gpu, tvec_gpu = pose_gpu

            # Rotation error: convert rvec to rotation matrices and compute angle
            R_cpu, _ = cv2.Rodrigues(rvec_cpu)
            R_gpu, _ = cv2.Rodrigues(rvec_gpu)
            from common.pose import rotation_error_deg
            rot_err_deg = rotation_error_deg(R_cpu, R_gpu)

            # Translation error
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
            num_quads_cpu,
            num_quads_gpu,
            t_cpu_quads,
            t_gpu_quads,
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
            "num_quads_cpu",
            "num_quads_gpu",
            "t_cpu_quads_ms",
            "t_gpu_quads_ms",
        ],
        rows,
    )

    # Compute summary statistics
    id_matches = [r[5] for r in rows]
    hamming_matches = [r[6] for r in rows]
    corner_rms_list = [r[7] for r in rows if r[7] != float('inf')]
    rot_err_list = [r[8] for r in rows if r[8] != float('inf')]
    trans_err_list = [r[9] for r in rows if r[9] != float('inf')]
    t_cpu_list = [r[12] for r in rows]
    t_gpu_list = [r[13] for r in rows]

    id_match_rate = sum(id_matches) / len(id_matches) if len(id_matches) > 0 else 0.0
    hamming_match_rate = sum(hamming_matches) / len(hamming_matches) if len(hamming_matches) > 0 else 0.0
    mean_corner_rms = np.mean(corner_rms_list) if len(corner_rms_list) > 0 else float('inf')
    max_corner_rms = np.max(corner_rms_list) if len(corner_rms_list) > 0 else float('inf')
    mean_rot_err = np.mean(rot_err_list) if len(rot_err_list) > 0 else float('inf')
    max_rot_err = np.max(rot_err_list) if len(rot_err_list) > 0 else float('inf')
    mean_trans_err = np.mean(trans_err_list) if len(trans_err_list) > 0 else float('inf')
    max_trans_err = np.max(trans_err_list) if len(trans_err_list) > 0 else float('inf')
    mean_cpu_quads = np.mean(t_cpu_list) if len(t_cpu_list) > 0 else 0.0
    mean_gpu_quads = np.mean(t_gpu_list) if len(t_gpu_list) > 0 else 0.0
    quads_speedup = mean_cpu_quads / mean_gpu_quads if mean_gpu_quads > 0 else 0.0

    # Write report
    report_lines = [
        "Phase 14C Validation: CPU vs GPU Quads in Full Pipeline (200 frames)",
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
        f"  mean_cpu_quads_ms: {mean_cpu_quads:.4f}",
        f"  mean_gpu_quads_ms: {mean_gpu_quads:.4f}",
        f"  quads_speedup: {quads_speedup:.4f}x",
        "",
        "Pass Criteria:",
        "  - id_match_rate >= 0.99",
        "  - hamming_match_rate >= 0.99",
        "  - mean_corner_rms_px <= 1.0",
        "  - max_corner_rms_px <= 2.5",
        "  - mean_rot_err_deg <= 0.5",
        "  - max_rot_err_deg <= 1.5",
        "  - mean_trans_err_m <= 0.01",
        "  - max_trans_err_m <= 0.02",
        "  - mean_gpu_quads_ms < mean_cpu_quads_ms (desired: >= 2.0x speedup)",
        "",
    ]

    # Check pass/fail
    passed = (
        id_match_rate >= 0.99
        and hamming_match_rate >= 0.99
        and mean_corner_rms <= 1.0
        and max_corner_rms <= 2.5
        and mean_rot_err <= 0.5
        and max_rot_err <= 1.5
        and mean_trans_err <= 0.01
        and max_trans_err <= 0.02
        and mean_gpu_quads < mean_cpu_quads
    )

    report_lines.append(f"Overall: {'PASS' if passed else 'FAIL'}")

    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Phase 14C validation complete: {csv_path}")
    print(f"Report: {report_path}")
    print(f"Overall: {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()

