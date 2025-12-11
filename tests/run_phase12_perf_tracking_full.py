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
from common.tracking import TagTrackerState, pose_delta, should_use_detect_mode, track_corners_klt
from common.video import load_frames
from gpu.corners import detect_quad_gpu
from gpu.decode import decode_gpu_bits, decode_gpu_codebook
from gpu.sampling import sample_gpu
from common.pose import solve_pose_gpu_from_homography
from cpu.detector import detect_apriltag
from tests.helpers import write_csv


def _now_ms():
    return time.perf_counter() * 1000.0


def run_frame_detect(idx, frame, gray, cfg, tracker):
    det = detect_apriltag(frame, cfg)
    if not det.detected or det.corners is None:
        tracker.has_valid = False
        return False
    corners_gpu = detect_quad_gpu(gray, cfg)
    if corners_gpu is None:
        tracker.has_valid = False
        return False
    gpu_sample = sample_gpu(gray, corners_gpu, cfg)
    gpu_bits = decode_gpu_bits(gpu_sample, cfg)
    decode_gpu_codebook(gpu_bits, cfg)
    solve_pose_gpu_from_homography(corners_gpu, gray.shape[:2], cfg)
    tracker.has_valid = True
    tracker.corners = corners_gpu
    tracker.last_frame_idx = idx
    return True


def run_frame_track(idx, gray, prev_gray, cfg, tracker):
    tracking_cfg = cfg.get("tracking", {})
    tracked_corners, ok_track = track_corners_klt(
        prev_gray,
        gray,
        tracker.corners,
        tracking_cfg.get("max_optical_flow_error", 10.0),
        tracking_cfg.get("min_tracked_points", 3),
    )
    if not ok_track:
        return False
    gpu_sample = sample_gpu(gray, tracked_corners, cfg)
    gpu_bits = decode_gpu_bits(gpu_sample, cfg)
    gpu_id, gpu_hamming = decode_gpu_codebook(gpu_bits, cfg)
    gpu_rvec, gpu_tvec = solve_pose_gpu_from_homography(tracked_corners, gray.shape[:2], cfg)
    if tracking_cfg.get("fallback_on_bad_decode", True):
        if gpu_id != cfg["tag"]["id_expected"] or gpu_hamming is None:
            return False
    if tracker.pose_rvec is not None and tracker.pose_tvec is not None:
        rot_delta, trans_delta = pose_delta(tracker.pose_rvec, tracker.pose_tvec, gpu_rvec, gpu_tvec)
        if rot_delta > tracking_cfg.get("max_rot_delta_deg", 10.0) or trans_delta > tracking_cfg.get("max_trans_delta_m", 0.2):
            return False
    tracker.has_valid = True
    tracker.corners = tracked_corners
    tracker.pose_rvec = gpu_rvec
    tracker.pose_tvec = gpu_tvec
    tracker.last_frame_idx = idx
    return True


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
    cfg["input"]["scale_factor"] = 1.0
    cfg["input"]["use_roi"] = False
    cfg["tracking"]["enabled"] = True

    frames = load_frames(cfg["input"]["video_path"], range(200))

    csv_path = Path("outputs/phase12_tracking_perf_full.csv")
    report_path = Path("outputs/phase12_tracking_perf_full_report.txt")

    tracker = TagTrackerState()
    prev_gray = None
    rows = []
    num_detect = 0
    num_track = 0
    num_fallback = 0

    for idx, frame in enumerate(frames):
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mode_used = "DETECT" if should_use_detect_mode(idx, tracker, cfg["tracking"]) else "TRACK"
        t0 = _now_ms()

        success = False
        if mode_used == "TRACK" and prev_gray is not None and tracker.has_valid:
            success = run_frame_track(idx, gray, prev_gray, cfg, tracker)
            if not success:
                num_fallback += 1
                mode_used = "DETECT"

        if mode_used == "DETECT":
            num_detect += 1
            success = run_frame_detect(idx, frame, gray, cfg, tracker)
        else:
            num_track += 1

        t1 = _now_ms()
        rows.append([idx, mode_used, t1 - t0])
        prev_gray = gray

    write_csv(csv_path, ["frame_index", "mode_used", "t_total_ms"], rows)

    total_vals = [r[2] for r in rows]
    stats = quantiles(total_vals)
    fps = 1000.0 / stats["mean"] if stats["mean"] > 0 else None

    baseline_mean = None
    baseline_fps = None
    baseline_report = Path("outputs/phase12_baseline_gpu_full_report.txt")
    if baseline_report.exists():
        for line in baseline_report.read_text(encoding="utf-8").splitlines():
            if line.startswith("mean_total_ms:"):
                baseline_mean = float(line.split(":")[1].strip())
            if line.startswith("fps:"):
                try:
                    baseline_fps = float(line.split(":")[1].strip())
                except ValueError:
                    baseline_fps = None

    lines = [
        f"frames_processed: {len(rows)}",
        f"num_detect_frames: {num_detect}",
        f"num_track_frames: {num_track}",
        f"num_fallback_frames: {num_fallback}",
        f"mean_total_ms: {stats['mean']}",
        f"median_total_ms: {stats['median']}",
        f"p90_total_ms: {stats['p90']}",
        f"p99_total_ms: {stats['p99']}",
        f"max_total_ms: {stats['max']}",
        f"fps_effective: {fps}",
    ]
    if baseline_mean is not None:
        lines.append(f"baseline_gpu_full_mean_total_ms: {baseline_mean}")
    if baseline_fps is not None:
        lines.append(f"baseline_gpu_full_fps: {baseline_fps}")
    if baseline_fps is not None and fps is not None:
        lines.append(f"speedup: {fps / baseline_fps}")
    if fps is not None and fps < 120:
        lines.append("WARNING: fps_effective below 120 FPS target.")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("Phase12 tracking perf complete:", csv_path)


if __name__ == "__main__":
    main()

