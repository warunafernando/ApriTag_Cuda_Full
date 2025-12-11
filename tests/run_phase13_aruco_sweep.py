from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import ensure_output_dirs, load_config
from common.pose import rotation_error_deg, solve_pose_cpu, solve_pose_gpu_from_homography, translation_error
from common.video import load_frames
from gpu.corners import detect_quad_gpu
from gpu.decode import decode_gpu_bits, decode_gpu_codebook
from gpu.sampling import sample_gpu
from cpu.detector import detect_apriltag
from cpu.decode import decode_apriltag
from tests.helpers import write_csv


PARAM_SETS = [
    {
        "name": "baseline",
        "params": {},
    },
    {
        "name": "fast1",
        "params": {
            "adaptiveThreshWinSizeMin": 3,
            "adaptiveThreshWinSizeMax": 15,
            "adaptiveThreshWinSizeStep": 6,
            "minMarkerPerimeterRate": 0.05,
            "maxMarkerPerimeterRate": 3.0,
            "cornerRefinementWinSize": 3,
            "cornerRefinementMaxIterations": 20,
        },
    },
    {
        "name": "fast2",
        "params": {
            "adaptiveThreshWinSizeMin": 3,
            "adaptiveThreshWinSizeMax": 11,
            "adaptiveThreshWinSizeStep": 4,
            "minMarkerPerimeterRate": 0.07,
            "maxMarkerPerimeterRate": 3.0,
            "cornerRefinementWinSize": 2,
            "cornerRefinementMaxIterations": 15,
        },
    },
]


def corner_rms(cpu_corners, gpu_corners):
    diff = cpu_corners.astype(np.float32) - gpu_corners.astype(np.float32)
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def quantiles(vals: list[float]):
    arr = np.array(vals, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def run_profile(profile, frames, cfg):
    cfg = dict(cfg)
    cfg["aruco_params"] = profile["params"]
    cfg["tracking"]["enabled"] = False
    perf_rows = []
    metrics_rows = []
    mean_totals = []
    corner_rms_vals = []
    rot_errs = []
    trans_errs = []
    id_matches = 0
    ham_matches = 0
    for idx, frame in enumerate(frames):
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cpu_det = detect_apriltag(frame, cfg)
        cpu_corners = cpu_det.corners if cpu_det.detected else None
        cpu_decode = decode_apriltag(gray, cpu_corners, cfg) if cpu_corners is not None else None
        cpu_pose = solve_pose_cpu(cpu_corners, gray.shape[:2], cfg) if cpu_corners is not None else (None, None)

        t0 = time.perf_counter() * 1000.0
        corners = detect_quad_gpu(gray, cfg)
        t_c1 = time.perf_counter() * 1000.0
        sample = sample_gpu(gray, corners, cfg) if corners is not None else None
        t_s1 = time.perf_counter() * 1000.0
        if sample is not None:
            bits = decode_gpu_bits(sample, cfg)
            gpu_id, gpu_ham = decode_gpu_codebook(bits, cfg)
        else:
            gpu_id = None
            gpu_ham = None
        t_d1 = time.perf_counter() * 1000.0
        gpu_pose = solve_pose_gpu_from_homography(corners, gray.shape[:2], cfg) if corners is not None else (None, None)
        t_p1 = time.perf_counter() * 1000.0
        t_total = t_p1 - t0

        perf_rows.append(
            [
                idx,
                t_c1 - t0,
                t_s1 - t_c1,
                t_d1 - t_s1,
                t_p1 - t_d1,
                t_total,
            ]
        )
        mean_totals.append(t_total)

        corner_rms_px = None
        rot_err = None
        trans_err = None
        if cpu_corners is not None and corners is not None:
            corner_rms_px = corner_rms(cpu_corners, corners)
            corner_rms_vals.append(corner_rms_px)
        if cpu_pose[0] is not None and gpu_pose[0] is not None:
            R_cpu, _ = cv2.Rodrigues(cpu_pose[0])
            R_gpu, _ = cv2.Rodrigues(gpu_pose[0])
            rot_err = rotation_error_deg(R_cpu, R_gpu)
            trans_err = translation_error(cpu_pose[1], gpu_pose[1])
            rot_errs.append(rot_err)
            trans_errs.append(trans_err)

        if cpu_decode and gpu_id is not None and cpu_decode.id == gpu_id:
            id_matches += 1
        if cpu_decode and gpu_ham is not None and cpu_decode.hamming == gpu_ham:
            ham_matches += 1

        metrics_rows.append(
            [
                idx,
                cpu_decode.id if cpu_decode else None,
                gpu_id,
                1 if cpu_decode and gpu_id is not None and cpu_decode.id == gpu_id else 0,
                corner_rms_px,
                rot_err,
                trans_err,
            ]
        )

    return {
        "perf_rows": perf_rows,
        "metrics_rows": metrics_rows,
        "mean_total": float(np.mean(mean_totals)),
        "mean_corner_rms": float(np.mean(corner_rms_vals)) if corner_rms_vals else None,
        "max_corner_rms": float(np.max(corner_rms_vals)) if corner_rms_vals else None,
        "id_match_rate": id_matches / len(frames),
        "mean_rot_err": float(np.mean(rot_errs)) if rot_errs else None,
        "max_rot_err": float(np.max(rot_errs)) if rot_errs else None,
        "mean_trans_err": float(np.mean(trans_errs)) if trans_errs else None,
        "max_trans_err": float(np.max(trans_errs)) if trans_errs else None,
    }


def main() -> None:
    cfg = load_config()
    ensure_output_dirs(cfg)
    frames = load_frames(cfg["input"]["video_path"], range(200))

    summary_lines = []
    best = None
    for profile in PARAM_SETS:
        result = run_profile(profile, frames, cfg)
        name = profile["name"]
        perf_csv = Path(f"outputs/phase13_aruco_{name}_perf.csv")
        metrics_csv = Path(f"outputs/phase13_aruco_{name}_metrics.csv")

        write_csv(
            perf_csv,
            ["frame_index", "t_corners_ms", "t_sampling_ms", "t_decode_ms", "t_pose_ms", "t_total_ms"],
            result["perf_rows"],
        )
        write_csv(
            metrics_csv,
            ["frame_index", "cpu_id", "gpu_id", "id_match_flag", "corner_rms_px", "rot_error_deg", "trans_error_m"],
            result["metrics_rows"],
        )

        valid = (
            result["id_match_rate"] == 1.0
            and (result["mean_corner_rms"] or 0) <= 0.6
            and (result["max_corner_rms"] or 0) <= 1.5
            and (result["max_rot_err"] or 0) < 3.0
            and (result["max_trans_err"] or 0) < 0.03
        )

        summary_lines.append(
            "\n".join(
                [
                    f"name: {name}",
                    f"mean_total_ms: {result['mean_total']}",
                    f"mean_corner_rms_px: {result['mean_corner_rms']}",
                    f"max_corner_rms_px: {result['max_corner_rms']}",
                    f"id_match_rate: {result['id_match_rate']}",
                    f"mean_rot_error_deg: {result['mean_rot_err']}",
                    f"max_rot_error_deg: {result['max_rot_err']}",
                    f"mean_trans_error_m: {result['mean_trans_err']}",
                    f"max_trans_error_m: {result['max_trans_err']}",
                    f"valid: {valid}",
                ]
            )
        )

        if valid:
            if best is None or result["mean_total"] < best["mean_total"]:
                best = {"name": name, "mean_total": result["mean_total"]}

    summary_path = Path("outputs/phase13_aruco_sweep_summary.txt")
    if best:
        summary_lines.append(f"best_valid_profile: {best['name']}")
        summary_lines.append(f"best_valid_mean_total_ms: {best['mean_total']}")
        summary_lines.append(f"best_valid_fps: {1000.0/best['mean_total']}")
    summary_path.write_text("\n\n".join(summary_lines), encoding="utf-8")
    print("Phase13 ArUco sweep complete:", summary_path)


if __name__ == "__main__":
    main()

