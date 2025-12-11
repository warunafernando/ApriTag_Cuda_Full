from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import ensure_output_dirs, load_config
from common.pose import rotation_error_deg, solve_pose_cpu, solve_pose_gpu_from_homography, translation_error
from common.tracking import TagTrackerState, pose_delta, should_use_detect_mode, track_corners_klt
from common.video import load_frames
from cpu.decode import decode_apriltag
from cpu.detector import detect_apriltag
from gpu.corners import detect_quad_gpu
from gpu.decode import decode_gpu_bits, decode_gpu_codebook
from gpu.sampling import sample_gpu
from tests.helpers import write_csv


def run_frame_detect(idx, frame, gray, cfg, tracker):
    detection = detect_apriltag(frame, cfg)
    if not detection.detected or detection.corners is None:
        tracker.has_valid = False
        return None, None, None, None, detection

    corners_gpu = detect_quad_gpu(gray, cfg)
    if corners_gpu is None:
        tracker.has_valid = False
        return None, None, None, None, detection

    gpu_sample = sample_gpu(gray, corners_gpu, cfg)
    gpu_bits = decode_gpu_bits(gpu_sample, cfg)
    gpu_id, gpu_hamming = decode_gpu_codebook(gpu_bits, cfg)
    gpu_rvec, gpu_tvec = solve_pose_gpu_from_homography(corners_gpu, gray.shape[:2], cfg)

    tracker.has_valid = True
    tracker.corners = corners_gpu
    tracker.pose_rvec = gpu_rvec
    tracker.pose_tvec = gpu_tvec
    tracker.last_frame_idx = idx
    return gpu_id, gpu_hamming, gpu_rvec, gpu_tvec, detection


def run_frame_track(idx, frame, gray, prev_gray, cfg, tracker):
    tracking_cfg = cfg.get("tracking", {})
    tracked_corners, ok_track = track_corners_klt(
        prev_gray,
        gray,
        tracker.corners,
        tracking_cfg.get("max_optical_flow_error", 10.0),
        tracking_cfg.get("min_tracked_points", 3),
    )
    if not ok_track:
        return None

    gpu_sample = sample_gpu(gray, tracked_corners, cfg)
    gpu_bits = decode_gpu_bits(gpu_sample, cfg)
    gpu_id, gpu_hamming = decode_gpu_codebook(gpu_bits, cfg)
    gpu_rvec, gpu_tvec = solve_pose_gpu_from_homography(tracked_corners, gray.shape[:2], cfg)

    # Validate decode and pose change vs last pose.
    if tracking_cfg.get("fallback_on_bad_decode", True):
        if gpu_id != cfg["tag"]["id_expected"] or gpu_hamming is None:
            return None
    if tracker.pose_rvec is not None and tracker.pose_tvec is not None:
        rot_delta, trans_delta = pose_delta(tracker.pose_rvec, tracker.pose_tvec, gpu_rvec, gpu_tvec)
        if rot_delta > tracking_cfg.get("max_rot_delta_deg", 10.0) or trans_delta > tracking_cfg.get("max_trans_delta_m", 0.2):
            return None

    # Accept
    tracker.has_valid = True
    tracker.corners = tracked_corners
    tracker.pose_rvec = gpu_rvec
    tracker.pose_tvec = gpu_tvec
    tracker.last_frame_idx = idx
    return gpu_id, gpu_hamming, gpu_rvec, gpu_tvec


def main() -> None:
    cfg = load_config()
    ensure_output_dirs(cfg)

    # For smoke, force detect_every_n=2 to ensure at least one TRACK.
    cfg["tracking"]["detect_every_n"] = 2
    frames = load_frames(cfg["input"]["video_path"], range(5))

    csv_path = Path("outputs/phase11_smoke5_tracking.csv")

    tracker = TagTrackerState()
    prev_gray = None
    rows = []

    for idx, frame in enumerate(frames):
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mode_used = "DETECT" if should_use_detect_mode(idx, tracker, cfg["tracking"]) else "TRACK"
        tracking_used = 1 if mode_used == "TRACK" else 0
        tracking_fallback = 0

        # CPU reference
        cpu_det = detect_apriltag(frame, cfg)
        cpu_decode = None
        cpu_pose = None
        if cpu_det.detected and cpu_det.corners is not None:
            cpu_decode = decode_apriltag(gray, cpu_det.corners, cfg)
            cpu_pose = solve_pose_cpu(cpu_det.corners, gray.shape[:2], cfg)

        if mode_used == "TRACK" and prev_gray is not None and tracker.has_valid:
            track_res = run_frame_track(idx, frame, gray, prev_gray, cfg, tracker)
            if track_res is None:
                tracking_fallback = 1
                mode_used = "DETECT"

        if mode_used == "DETECT":
            gpu_id, gpu_hamming, gpu_rvec, gpu_tvec, cpu_det_used = run_frame_detect(idx, frame, gray, cfg, tracker)
        else:
            gpu_id, gpu_hamming, gpu_rvec, gpu_tvec = track_res  # type: ignore
            cpu_det_used = cpu_det

        rot_err = None
        trans_err = None
        if cpu_pose is not None and gpu_rvec is not None and gpu_tvec is not None:
            R_cpu, _ = cv2.Rodrigues(cpu_pose[0])
            R_gpu, _ = cv2.Rodrigues(gpu_rvec)
            rot_err = rotation_error_deg(R_cpu, R_gpu)
            trans_err = translation_error(cpu_pose[1], gpu_tvec)

        rows.append(
            [
                idx,
                mode_used,
                cpu_decode.id if cpu_decode else None,
                gpu_id,
                cpu_decode.hamming if cpu_decode else None,
                gpu_hamming,
                rot_err,
                trans_err,
                tracking_used,
                tracking_fallback,
            ]
        )
        prev_gray = gray

    write_csv(
        csv_path,
        [
            "frame_index",
            "mode_used",
            "cpu_id",
            "gpu_id",
            "cpu_hamming",
            "gpu_hamming",
            "rot_error_deg",
            "trans_error_m",
            "tracking_used",
            "tracking_fallback",
        ],
        rows,
    )
    print("Phase11 smoke tracking complete:", csv_path)


if __name__ == "__main__":
    main()

