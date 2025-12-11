"""
Phase 14D_4: Parameter tuning sweep for GPU corner refinement accuracy.
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
from common.pose import rotation_error_deg, solve_pose_cpu
from common.video import load_frames
from cpu.corner_refine import cpu_refine_corners
from cpu.decode import decode_apriltag
from cpu.edges import cpu_canny_edges
from cpu.quads import cpu_quad_candidates_from_edges
from tests.helpers import rms

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


def test_config(cfg, frames, num_frames=50):
    """Test a configuration and return metrics."""
    edges_cfg = cfg.get("edges", {})
    low_thresh = int(edges_cfg.get("low_thresh", 35))
    high_thresh = int(edges_cfg.get("high_thresh", 110))
    expected_id = cfg.get("tag", {}).get("id_expected", 3)

    corner_rms_list = []
    rot_err_list = []
    trans_err_list = []
    timings_list = []

    for idx in range(min(num_frames, len(frames))):
        frame = frames[idx]
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges_cpu = cpu_canny_edges(gray, low_thresh=low_thresh, high_thresh=high_thresh)
        quads_cpu = cpu_quad_candidates_from_edges(edges_cpu, cfg)
        quad_best = select_best_quad(quads_cpu)

        if quad_best is None:
            continue

        corners_in = quad_best.reshape(1, 4, 2).astype(np.float32)
        image_shape = (gray.shape[0], gray.shape[1])

        # CPU reference
        cfg_cpu = cfg.copy()
        cfg_cpu["corner_refine"] = cfg["corner_refine"].copy()
        cfg_cpu["corner_refine"]["mode"] = "CPU"
        corners_cpu, _ = dispatch_corner_refine(gray, corners_in, cfg_cpu, image_shape, num_tags=1)
        corners_cpu = corners_cpu[0]
        decode_cpu = decode_apriltag(gray, corners_cpu, cfg)
        pose_cpu = solve_pose_cpu(corners_cpu, image_shape, cfg)

        # GPU test
        corners_gpu, timings = dispatch_corner_refine(gray, corners_in, cfg, image_shape, num_tags=1)
        corners_gpu = corners_gpu[0]

        corner_rms = rms(corners_cpu, corners_gpu)
        corner_rms_list.append(corner_rms)

        if pose_cpu is not None:
            from gpu.corner_refine import gpu_refine_corners
            from common.pose import solve_pose_gpu_from_homography
            pose_gpu = solve_pose_gpu_from_homography(corners_gpu, image_shape, cfg)
            if pose_gpu is not None:
                rvec_cpu, tvec_cpu = pose_cpu
                rvec_gpu, tvec_gpu = pose_gpu
                R_cpu, _ = cv2.Rodrigues(rvec_cpu)
                R_gpu, _ = cv2.Rodrigues(rvec_gpu)
                rot_err = rotation_error_deg(R_cpu, R_gpu)
                trans_err = float(np.linalg.norm(tvec_cpu - tvec_gpu))
                rot_err_list.append(rot_err)
                trans_err_list.append(trans_err)

        refine_ms = timings.get("t_refine_ms", 0.0)
        if refine_ms > 0:
            timings_list.append(refine_ms)

    mean_rms = np.mean(corner_rms_list) if len(corner_rms_list) > 0 else float('inf')
    mean_rot = np.mean(rot_err_list) if len(rot_err_list) > 0 else float('inf')
    mean_trans = np.mean(trans_err_list) if len(trans_err_list) > 0 else float('inf')
    mean_time = np.mean(timings_list) if len(timings_list) > 0 else 0.0

    return mean_rms, mean_rot, mean_trans, mean_time


def main() -> None:
    cfg = load_config()
    ensure_output_dirs(cfg)

    video_path = cfg["input"]["video_path"]
    frames = load_frames(video_path, range(50))  # Use 50 frames for tuning

    # Parameter combinations to test
    configs = []

    # Base config
    base_cfg = cfg.copy()
    base_cfg["corner_refine"] = cfg["corner_refine"].copy()
    base_cfg["corner_refine"]["mode"] = "GPU"
    base_cfg["corner_refine"]["allow_failover"] = False
    base_cfg["corner_refine"]["threads_per_block"] = 256  # Best from block sweep

    # Test different parameter combinations
    test_params = [
        {"max_iters": 3, "epsilon": 0.05, "step_max": 0.5, "det_min": 1e-4, "grad_thresh": 1.0},
        {"max_iters": 5, "epsilon": 0.03, "step_max": 0.5, "det_min": 1e-4, "grad_thresh": 1.0},
        {"max_iters": 5, "epsilon": 0.03, "step_max": 1.0, "det_min": 1e-4, "grad_thresh": 1.0},
        {"max_iters": 7, "epsilon": 0.02, "step_max": 0.5, "det_min": 1e-4, "grad_thresh": 1.0},
        {"max_iters": 5, "epsilon": 0.03, "step_max": 0.5, "det_min": 1e-5, "grad_thresh": 0.5},
        {"max_iters": 5, "epsilon": 0.03, "step_max": 0.5, "det_min": 1e-4, "grad_thresh": 0.5},
    ]

    results = []

    for params in test_params:
        test_cfg = base_cfg.copy()
        test_cfg["corner_refine"].update(params)
        print(f"Testing: {params}")
        mean_rms, mean_rot, mean_trans, mean_time = test_config(test_cfg, frames, num_frames=50)
        results.append((params, mean_rms, mean_rot, mean_trans, mean_time))
        print(f"  RMS: {mean_rms:.4f}, Rot: {mean_rot:.4f}°, Trans: {mean_trans:.4f}m, Time: {mean_time:.4f}ms")
        print()

    # Write tuning log
    output_path = Path("outputs/phase14D3_tuning_log.txt")
    lines = [
        "Phase 14D_4 GPU Corner Refinement Parameter Tuning",
        "=" * 70,
        "",
        "Tested configurations (50 frames each):",
        "",
    ]

    for params, mean_rms, mean_rot, mean_trans, mean_time in results:
        lines.append(f"Config: {params}")
        lines.append(f"  mean_corner_rms_px: {mean_rms:.4f}")
        lines.append(f"  mean_rot_err_deg: {mean_rot:.4f}")
        lines.append(f"  mean_trans_err_m: {mean_trans:.4f}")
        lines.append(f"  mean_refine_ms: {mean_time:.4f}")
        lines.append("")

    # Find best config (lowest RMS that meets targets)
    valid_results = [
        r for r in results
        if r[1] <= 0.25 and r[2] <= 0.5 and r[3] <= 0.008
    ]

    if len(valid_results) > 0:
        best = min(valid_results, key=lambda x: x[1])  # Lowest RMS
        lines.extend([
            "Best Configuration (meets all targets):",
            f"  {best[0]}",
            f"  mean_corner_rms_px: {best[1]:.4f}",
            f"  mean_rot_err_deg: {best[2]:.4f}",
            f"  mean_trans_err_m: {best[3]:.4f}",
            f"  mean_refine_ms: {best[4]:.4f}",
        ])
    else:
        best = min(results, key=lambda x: x[1])  # Lowest RMS overall
        lines.extend([
            "Best Configuration (lowest RMS, may not meet all targets):",
            f"  {best[0]}",
            f"  mean_corner_rms_px: {best[1]:.4f}",
            f"  mean_rot_err_deg: {best[2]:.4f}",
            f"  mean_trans_err_m: {best[3]:.4f}",
            f"  mean_refine_ms: {best[4]:.4f}",
        ])

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Tuning sweep complete: {output_path}")


if __name__ == "__main__":
    main()

