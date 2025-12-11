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
from cpu.decode import decode_apriltag
from cpu.detector import detect_apriltag
from gpu.corners import detect_quad_gpu
from gpu.decode import decode_gpu_bits, decode_gpu_codebook
from gpu.sampling import sample_gpu
from tests.helpers import write_csv


def _now_ms():
    return time.perf_counter() * 1000.0


def corner_rms(cpu_corners, gpu_corners):
    diff = cpu_corners.astype(np.float32) - gpu_corners.astype(np.float32)
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def quantiles(vals: list[float]):
    arr = np.array(vals, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def main() -> None:
    cfg = load_config()
    # Enforce baseline constraints
    cfg["tracking"]["enabled"] = False
    cfg["input"]["scale_factor"] = 1.0
    cfg["input"]["use_roi"] = False

    ensure_output_dirs(cfg)
    frames = load_frames(cfg["input"]["video_path"], range(200))

    csv_path = Path("outputs/phase12_baseline_gpu_full.csv")
    report_path = Path("outputs/phase12_baseline_gpu_full_report.txt")

    rows = []
    corner_rms_vals = []
    rot_err_vals = []
    trans_err_vals = []
    id_matches = 0
    ham_matches = 0
    cpu_detected = 0
    gpu_detected = 0

    for idx, frame in enumerate(frames):
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cpu_det = detect_apriltag(frame, cfg)
        cpu_id = cpu_det.id if cpu_det.detected else None
        cpu_ham = None
        cpu_corners = cpu_det.corners if cpu_det.detected else None
        cpu_pose = None
        if cpu_det.detected and cpu_corners is not None:
            cpu_detected += 1
            cpu_decode = decode_apriltag(gray, cpu_corners, cfg)
            cpu_ham = cpu_decode.hamming
            cpu_id = cpu_decode.id
            cpu_pose = solve_pose_cpu(cpu_corners, gray.shape[:2], cfg)

        t_corners0 = _now_ms()
        gpu_corners = detect_quad_gpu(gray, cfg)
        t_corners1 = _now_ms()
        t_sampling0 = _now_ms()
        gpu_sample = sample_gpu(gray, gpu_corners, cfg) if gpu_corners is not None else None
        t_sampling1 = _now_ms()
        t_decode0 = _now_ms()
        if gpu_sample is not None:
            gpu_bits = decode_gpu_bits(gpu_sample, cfg)
            gpu_id, gpu_ham = decode_gpu_codebook(gpu_bits, cfg)
        else:
            gpu_id = None
            gpu_ham = None
        t_decode1 = _now_ms()
        t_pose0 = _now_ms()
        gpu_pose = solve_pose_gpu_from_homography(gpu_corners, gray.shape[:2], cfg) if gpu_corners is not None else (None, None)
        t_pose1 = _now_ms()

        t_total = t_pose1 - t_corners0

        corner_rms_px = None
        rot_err = None
        trans_err = None
        if cpu_corners is not None and gpu_corners is not None:
            gpu_detected += 1
            corner_rms_px = corner_rms(cpu_corners, gpu_corners)
            corner_rms_vals.append(corner_rms_px)
        if cpu_pose is not None and gpu_pose[0] is not None and gpu_pose[1] is not None:
            R_cpu, _ = cv2.Rodrigues(cpu_pose[0])
            R_gpu, _ = cv2.Rodrigues(gpu_pose[0])
            rot_err = rotation_error_deg(R_cpu, R_gpu)
            trans_err = translation_error(cpu_pose[1], gpu_pose[1])
            rot_err_vals.append(rot_err)
            trans_err_vals.append(trans_err)

        if cpu_id is not None and gpu_id is not None and cpu_id == gpu_id:
            id_matches += 1
        if cpu_ham is not None and gpu_ham is not None and cpu_ham == gpu_ham:
            ham_matches += 1

        rows.append(
            [
                idx,
                cpu_id,
                gpu_id,
                cpu_ham,
                gpu_ham,
                1 if (cpu_id is not None and cpu_id == gpu_id) else 0,
                1 if (cpu_ham is not None and gpu_ham is not None and cpu_ham == gpu_ham) else 0,
                corner_rms_px,
                rot_err,
                trans_err,
                0.0,  # t_gpu_copy_in_ms placeholder
                0.0,  # t_gpu_preprocess_ms placeholder
                t_corners1 - t_corners0,
                t_sampling1 - t_sampling0,
                t_decode1 - t_decode0,
                t_pose1 - t_pose0,
                t_total,
            ]
        )

    header = [
        "frame_index",
        "cpu_id",
        "gpu_id",
        "cpu_hamming",
        "gpu_hamming",
        "id_match_flag",
        "hamming_match_flag",
        "corner_rms_px",
        "rot_error_deg",
        "trans_error_m",
        "t_gpu_copy_in_ms",
        "t_gpu_preprocess_ms",
        "t_gpu_corners_ms",
        "t_gpu_sampling_ms",
        "t_gpu_decode_ms",
        "t_gpu_pose_ms",
        "t_gpu_total_ms",
    ]
    write_csv(csv_path, header, rows)

    total_vals = [r[16] for r in rows]
    stats_total = quantiles(total_vals)
    fps = 1000.0 / stats_total["mean"] if stats_total["mean"] > 0 else None

    corner_stats = quantiles(corner_rms_vals) if corner_rms_vals else None
    rot_stats = quantiles(rot_err_vals) if rot_err_vals else None
    trans_stats = quantiles(trans_err_vals) if trans_err_vals else None

    corner_pass = corner_stats is not None and corner_stats["mean"] < 1.0 and corner_stats["max"] < 2.0

    report_lines = [
        f"frames_processed: {len(rows)}",
        f"cpu_detection_rate: {cpu_detected/len(rows)}",
        f"gpu_detection_rate: {gpu_detected/len(rows)}",
        f"id_match_rate: {id_matches/len(rows)}",
        f"hamming_match_rate: {ham_matches/len(rows)}",
        f"mean_total_ms: {stats_total['mean']}",
        f"median_total_ms: {stats_total['median']}",
        f"p90_total_ms: {stats_total['p90']}",
        f"p99_total_ms: {stats_total['p99']}",
        f"max_total_ms: {stats_total['max']}",
        f"fps: {fps}",
    ]
    if corner_stats:
        report_lines += [
            f"mean_corner_rms_px: {corner_stats['mean']}",
            f"median_corner_rms_px: {corner_stats['median']}",
            f"p90_corner_rms_px: {corner_stats['p90']}",
            f"max_corner_rms_px: {corner_stats['max']}",
            f"corner_rms_pass: {corner_pass}",
        ]
    if rot_stats and trans_stats:
        report_lines += [
            f"mean_rot_error_deg: {rot_stats['mean']}",
            f"max_rot_error_deg: {rot_stats['max']}",
            f"mean_trans_error_m: {trans_stats['mean']}",
            f"max_trans_error_m: {trans_stats['max']}",
        ]

    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print("Phase12 baseline GPU full complete:", csv_path)


if __name__ == "__main__":
    main()

