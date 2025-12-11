"""
Phase 14D_3: Compare CPU vs GPU vs HYBRID corner refinement accuracy (200 frames).
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
from common.pose import rotation_error_deg, solve_pose_cpu, solve_pose_gpu_from_homography
from common.video import load_frames
from cpu.decode import decode_apriltag
from cpu.edges import cpu_canny_edges
from cpu.quads import cpu_quad_candidates_from_edges
from gpu.decode import decode_gpu_bits, decode_gpu_codebook
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


def run_mode(
    gray: np.ndarray,
    corners_in: np.ndarray,
    cfg: dict,
    mode: str,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, dict]:
    """Run refinement in specified mode."""
    cfg_copy = cfg.copy()
    cfg_copy["corner_refine"] = cfg["corner_refine"].copy()
    cfg_copy["corner_refine"]["mode"] = mode
    return dispatch_corner_refine(gray, corners_in, cfg_copy, image_shape, num_tags=1)


def main() -> None:
    cfg = load_config()
    ensure_output_dirs(cfg)

    video_path = cfg["input"]["video_path"]
    frames = load_frames(video_path, range(200))

    edges_cfg = cfg.get("edges", {})
    low_thresh = int(edges_cfg.get("low_thresh", 35))
    high_thresh = int(edges_cfg.get("high_thresh", 110))
    expected_id = cfg.get("tag", {}).get("id_expected", 3)

    modes = ["CPU", "GPU", "HYBRID"]
    results = {mode: [] for mode in modes}

    for idx, frame in enumerate(frames):
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges_cpu = cpu_canny_edges(gray, low_thresh=low_thresh, high_thresh=high_thresh)
        quads_cpu = cpu_quad_candidates_from_edges(edges_cpu, cfg)
        quad_best = select_best_quad(quads_cpu)

        if quad_best is None:
            for mode in modes:
                results[mode].append([idx, None, None, None, None, None, None, None, None, None])
            continue

        corners_in = quad_best.reshape(1, 4, 2).astype(np.float32)
        image_shape = (gray.shape[0], gray.shape[1])

        # Reference: CPU mode
        corners_cpu, _ = run_mode(gray, corners_in, cfg, "CPU", image_shape)
        corners_cpu = corners_cpu[0]
        decode_cpu = decode_apriltag(gray, corners_cpu, cfg)
        pose_cpu = solve_pose_cpu(corners_cpu, image_shape, cfg)

        for mode in modes:
            if mode == "CPU":
                corners_refined = corners_cpu
                decode_result = decode_cpu
                pose = pose_cpu
                timings = {"t_refine_ms": 0.0}
            else:
                corners_refined, timings = run_mode(gray, corners_in, cfg, mode, image_shape)
                corners_refined = corners_refined[0]

                # Decode
                sample_grid_gpu = sample_gpu(gray, corners_refined, cfg)
                sample_grid_cpu = sample_grid_gpu if isinstance(sample_grid_gpu, np.ndarray) else cp.asnumpy(sample_grid_gpu)
                gpu_bits = decode_gpu_bits(sample_grid_cpu, cfg)
                id_gpu, hamming_gpu = decode_gpu_codebook(gpu_bits, cfg)
                decode_result = type('obj', (object,), {'id': id_gpu, 'hamming': hamming_gpu})()

                # Pose
                pose = solve_pose_gpu_from_homography(corners_refined, image_shape, cfg)

            # Compare with CPU reference
            corner_rms = rms(corners_cpu, corners_refined)

            id_match = (
                decode_result.id == decode_cpu.id == expected_id
                if (decode_result.id is not None and decode_cpu.id is not None)
                else False
            )
            hamming_match = (
                decode_result.hamming == decode_cpu.hamming == 0
                if (decode_result.hamming is not None and decode_cpu.hamming is not None)
                else False
            )

            if pose_cpu is not None and pose is not None:
                rvec_cpu, tvec_cpu = pose_cpu
                rvec_mode, tvec_mode = pose
                R_cpu, _ = cv2.Rodrigues(rvec_cpu)
                R_mode, _ = cv2.Rodrigues(rvec_mode)
                rot_err = rotation_error_deg(R_cpu, R_mode)
                trans_err = float(np.linalg.norm(tvec_cpu - tvec_mode))
            else:
                rot_err = float('inf')
                trans_err = float('inf')

            results[mode].append([
                idx,
                decode_result.id,
                decode_result.hamming,
                id_match,
                hamming_match,
                corner_rms,
                rot_err,
                trans_err,
                timings.get("t_refine_ms", 0.0),
            ])

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/200 frames...")

    # Write CSVs and compute summaries
    summary_lines = [
        "Phase 14D_3 Corner Refinement Comparison (200 frames)",
        "=" * 70,
        "",
    ]

    for mode in modes:
        csv_path = Path(f"outputs/phase14D3_compare_200_{mode.lower()}.csv")
        write_csv(
            csv_path,
            [
                "frame_index",
                "id",
                "hamming",
                "id_match",
                "hamming_match",
                "corner_rms_px",
                "rot_err_deg",
                "trans_err_m",
                "t_refine_ms",
            ],
            results[mode],
        )

        # Compute summary stats
        id_matches = [r[3] for r in results[mode] if r[3] is not None]
        hamming_matches = [r[4] for r in results[mode] if r[4] is not None]
        corner_rms_list = [r[5] for r in results[mode] if r[5] != float('inf')]
        rot_err_list = [r[6] for r in results[mode] if r[6] != float('inf')]
        trans_err_list = [r[7] for r in results[mode] if r[7] != float('inf')]
        t_refine_list = [r[8] for r in results[mode] if r[8] > 0]

        id_match_rate = sum(id_matches) / len(id_matches) if len(id_matches) > 0 else 0.0
        hamming_match_rate = sum(hamming_matches) / len(hamming_matches) if len(hamming_matches) > 0 else 0.0
        mean_corner_rms = np.mean(corner_rms_list) if len(corner_rms_list) > 0 else float('inf')
        max_corner_rms = np.max(corner_rms_list) if len(corner_rms_list) > 0 else float('inf')
        mean_rot_err = np.mean(rot_err_list) if len(rot_err_list) > 0 else float('inf')
        max_rot_err = np.max(rot_err_list) if len(rot_err_list) > 0 else float('inf')
        mean_trans_err = np.mean(trans_err_list) if len(trans_err_list) > 0 else float('inf')
        max_trans_err = np.max(trans_err_list) if len(trans_err_list) > 0 else float('inf')
        mean_refine_ms = np.mean(t_refine_list) if len(t_refine_list) > 0 else 0.0

        summary_lines.extend([
            f"{mode} Mode:",
            f"  id_match_rate: {id_match_rate:.4f}",
            f"  hamming_match_rate: {hamming_match_rate:.4f}",
            f"  mean_corner_rms_px: {mean_corner_rms:.4f}",
            f"  max_corner_rms_px: {max_corner_rms:.4f}",
            f"  mean_rot_err_deg: {mean_rot_err:.4f}",
            f"  max_rot_err_deg: {max_rot_err:.4f}",
            f"  mean_trans_err_m: {mean_trans_err:.4f}",
            f"  max_trans_err_m: {max_trans_err:.4f}",
            f"  mean_refine_ms: {mean_refine_ms:.4f}",
            "",
        ])

    # Write summary report
    report_path = Path("outputs/phase14D3_compare_200_report.txt")
    report_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Phase 14D_3 comparison complete: {report_path}")


if __name__ == "__main__":
    main()

