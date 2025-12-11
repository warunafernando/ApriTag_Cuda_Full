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
from common.video import load_frames
from cpu.detector import detect_apriltag
from tests.helpers import save_overlay, write_csv


PASS_ROT = 1.5
PASS_TRANS = 0.015
NOTIFY_ROT = 3.0
NOTIFY_TRANS = 0.03


def draw_pose(frame, rvec, tvec, K):
    if rvec is None or tvec is None:
        return frame
    axis_len = 0.05
    axis = np.float32([[axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]]).reshape(-1, 3)
    dist = np.zeros(5, dtype=np.float64)
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, K, dist)
    imgpts = imgpts.reshape(-1, 2).astype(int)
    img = frame.copy()
    origin = tuple(imgpts[2])  # z axis end as origin fallback if needed
    cv2.line(img, origin, tuple(imgpts[0]), (255, 255, 0), 2)  # X cyan-ish
    cv2.line(img, origin, tuple(imgpts[1]), (255, 255, 0), 2)  # Y cyan-ish
    cv2.line(img, origin, tuple(imgpts[2]), (255, 255, 0), 2)  # Z cyan-ish
    return img


def draw_pose_both(frame, cpu_rvec, cpu_tvec, gpu_rvec, gpu_tvec, K):
    dist = np.zeros(5, dtype=np.float64)
    axis_len = 0.05
    axis = np.float32([[axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]]).reshape(-1, 3)
    img = frame.copy()
    if cpu_rvec is not None and cpu_tvec is not None:
        imgpts, _ = cv2.projectPoints(axis, cpu_rvec, cpu_tvec, K, dist)
        imgpts = imgpts.reshape(-1, 2).astype(int)
        origin = tuple(imgpts[2])
        cv2.line(img, origin, tuple(imgpts[0]), (255, 255, 0), 2)  # cyan CPU
        cv2.line(img, origin, tuple(imgpts[1]), (255, 255, 0), 2)
        cv2.line(img, origin, tuple(imgpts[2]), (255, 255, 0), 2)
    if gpu_rvec is not None and gpu_tvec is not None:
        imgpts, _ = cv2.projectPoints(axis, gpu_rvec, gpu_tvec, K, dist)
        imgpts = imgpts.reshape(-1, 2).astype(int)
        origin = tuple(imgpts[2])
        cv2.line(img, origin, tuple(imgpts[0]), (255, 0, 255), 2)  # magenta GPU
        cv2.line(img, origin, tuple(imgpts[1]), (255, 0, 255), 2)
        cv2.line(img, origin, tuple(imgpts[2]), (255, 0, 255), 2)
    return img


def main() -> None:
    cfg = load_config()
    ensure_output_dirs(cfg)

    frames = load_frames(cfg["input"]["video_path"], range(5))
    csv_path = Path("outputs/phase6_smoke5_pose.csv")
    debug_dir = Path("outputs/debug_phase6")

    rows = []
    for idx, frame in enumerate(frames):
        detection = detect_apriltag(frame, cfg)
        if not detection.detected or detection.corners is None:
            rows.append([idx, False, None, None, None, None, None])
            continue

        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[:2]

        cpu_rvec, cpu_tvec = solve_pose_cpu(detection.corners, img_shape, cfg)
        gpu_rvec, gpu_tvec = solve_pose_gpu_from_homography(detection.corners, img_shape, cfg)

        if cpu_rvec is None or gpu_rvec is None:
            rows.append([idx, True, False, None, None, None, None])
            continue

        R_cpu, _ = cv2.Rodrigues(cpu_rvec)
        R_gpu, _ = cv2.Rodrigues(gpu_rvec)
        rot_err = rotation_error_deg(R_cpu, R_gpu)
        trans_err = translation_error(cpu_tvec, gpu_tvec)

        status = "PASS"
        if rot_err >= 3.0 or trans_err >= 0.03:
            status = "FAIL"
        elif rot_err >= 1.5 or trans_err >= 0.015:
            status = "NOTIFY"

        K, _ = _camera_from_shape(img_shape)
        overlay = draw_pose_both(frame, cpu_rvec, cpu_tvec, gpu_rvec, gpu_tvec, K)
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / f"frame_{idx:03d}_pose_overlay.png"), overlay)

        rows.append([idx, True, True, rot_err, trans_err, status, detection.id])

    write_csv(
        csv_path,
        ["frame", "cpu_detected", "gpu_pose_ok", "rot_error_deg", "trans_error_m", "status", "id"],
        rows,
    )
    rot_vals = [r[3] for r in rows if r[3] is not None]
    trans_vals = [r[4] for r in rows if r[4] is not None]
    print(
        f"Phase6 smoke complete. rot_mean={np.mean(rot_vals) if rot_vals else None} trans_mean={np.mean(trans_vals) if trans_vals else None}"
    )


def _camera_from_shape(shape):
    h, w = shape
    focal = max(h, w)
    K = np.array([[focal, 0, w / 2.0], [0, focal, h / 2.0], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)
    return K, dist


if __name__ == "__main__":
    main()

