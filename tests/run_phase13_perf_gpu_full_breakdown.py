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
from common.pose import solve_pose_gpu_from_homography
from common.video import load_frames
from gpu.corners import detect_quad_gpu
from gpu.decode import decode_gpu_bits, decode_gpu_codebook
from gpu.sampling import sample_gpu
from cpu.detector import detect_apriltag
from tests.helpers import write_csv


def _now_ms():
    return time.perf_counter() * 1000.0


def quantiles(vals):
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
    ensure_output_dirs(cfg)
    cfg["tracking"]["enabled"] = False
    cfg["input"]["scale_factor"] = 1.0
    cfg["input"]["use_roi"] = False

    frames = load_frames(cfg["input"]["video_path"], range(200))

    csv_path = Path("outputs/phase13_gpu_full_breakdown.csv")
    report_path = Path("outputs/phase13_gpu_full_breakdown_report.txt")

    rows = []
    total_vals = []
    copy_vals = []
    prep_vals = []
    corner_vals = []
    sampling_vals = []
    decode_vals = []
    pose_vals = []

    for idx, frame in enumerate(frames):
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        t0 = _now_ms()
        # Copy/preprocess placeholders (grayscale already)
        t_copy = 0.0
        t_pre = 0.0

        t_c0 = _now_ms()
        corners = detect_quad_gpu(gray, cfg)
        t_c1 = _now_ms()

        t_s0 = _now_ms()
        sample = sample_gpu(gray, corners, cfg) if corners is not None else None
        t_s1 = _now_ms()

        t_d0 = _now_ms()
        if sample is not None:
            bits = decode_gpu_bits(sample, cfg)
            decode_gpu_codebook(bits, cfg)
        t_d1 = _now_ms()

        t_p0 = _now_ms()
        if corners is not None:
            solve_pose_gpu_from_homography(corners, gray.shape[:2], cfg)
        t_p1 = _now_ms()

        t1 = t_p1

        t_corners = t_c1 - t_c0
        t_sampling = t_s1 - t_s0
        t_decode = t_d1 - t_d0
        t_pose = t_p1 - t_p0
        t_total = t1 - t0

        rows.append(
            [
                idx,
                t_copy,
                t_pre,
                t_corners,
                t_sampling,
                t_decode,
                t_pose,
                t_total,
            ]
        )
        copy_vals.append(t_copy)
        prep_vals.append(t_pre)
        corner_vals.append(t_corners)
        sampling_vals.append(t_sampling)
        decode_vals.append(t_decode)
        pose_vals.append(t_pose)
        total_vals.append(t_total)

    header = [
        "frame_index",
        "t_gpu_copy_in_ms",
        "t_gpu_preprocess_ms",
        "t_gpu_corners_ms",
        "t_gpu_sampling_ms",
        "t_gpu_decode_ms",
        "t_gpu_pose_ms",
        "t_gpu_total_ms",
    ]
    write_csv(csv_path, header, rows)

    stats_total = quantiles(total_vals)
    stats_corners = quantiles(corner_vals)
    stats_sampling = quantiles(sampling_vals)
    stats_decode = quantiles(decode_vals)
    stats_pose = quantiles(pose_vals)

    report_lines = [
        f"frames_processed: {len(rows)}",
        f"mean_copy_in_ms: {np.mean(copy_vals):.4f}",
        f"mean_preprocess_ms: {np.mean(prep_vals):.4f}",
        f"mean_corners_ms: {stats_corners['mean']}",
        f"mean_sampling_ms: {stats_sampling['mean']}",
        f"mean_decode_ms: {stats_decode['mean']}",
        f"mean_pose_ms: {stats_pose['mean']}",
        f"mean_total_ms: {stats_total['mean']}",
        f"median_total_ms: {stats_total['median']}",
        f"p90_total_ms: {stats_total['p90']}",
        f"p99_total_ms: {stats_total['p99']}",
        f"max_total_ms: {stats_total['max']}",
        f"fps_total: {1000.0/stats_total['mean'] if stats_total['mean']>0 else None}",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print("Phase13 breakdown complete:", csv_path)


if __name__ == "__main__":
    main()

