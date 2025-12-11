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
from tests.helpers import write_csv


def main() -> None:
    cfg = load_config()
    ensure_output_dirs(cfg)

    frames = load_frames(cfg["input"]["video_path"], range(200))
    csv_path = Path("outputs/phase6_validate200_pose.csv")
    report_path = Path("outputs/phase6_report.txt")
    debug_dir = Path("outputs/debug_phase6")

    rows = []
    rot_list = []
    trans_list = []
    fail_frames = 0

    for idx, frame in enumerate(frames):
        detection = detect_apriltag(frame, cfg)
        if not detection.detected or detection.corners is None:
            rows.append([idx, False, None, None, None, None, detection.id if detection.id else None])
            fail_frames += 1
            continue

        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[:2]

        cpu_rvec, cpu_tvec = solve_pose_cpu(detection.corners, img_shape, cfg)
        gpu_rvec, gpu_tvec = solve_pose_gpu_from_homography(detection.corners, img_shape, cfg)

        if cpu_rvec is None or gpu_rvec is None:
            rows.append([idx, True, False, None, None, "FAIL", detection.id])
            fail_frames += 1
            continue

        R_cpu, _ = cv2.Rodrigues(cpu_rvec)
        R_gpu, _ = cv2.Rodrigues(gpu_rvec)
        rot_err = rotation_error_deg(R_cpu, R_gpu)
        trans_err = translation_error(cpu_tvec, gpu_tvec)
        rot_list.append(rot_err)
        trans_list.append(trans_err)

        status = "PASS"
        if rot_err >= 3.0 or trans_err >= 0.03:
            status = "FAIL"
            fail_frames += 1
        elif rot_err >= 1.5 or trans_err >= 0.015:
            status = "NOTIFY"

        rows.append([idx, True, True, rot_err, trans_err, status, detection.id])

    write_csv(
        csv_path,
        ["frame", "cpu_detected", "gpu_pose_ok", "rot_error_deg", "trans_error_m", "status", "id"],
        rows,
    )

    mean_rot = float(np.mean(rot_list)) if rot_list else None
    mean_trans = float(np.mean(trans_list)) if trans_list else None

    total = len(frames)
    report_lines = [
        f"frames_processed: {total}",
        f"cpu_detected: {sum(1 for r in rows if r[1])}",
        f"gpu_pose_ok: {sum(1 for r in rows if r[2])}",
        f"mean_rot: {mean_rot}",
        f"mean_trans: {mean_trans}",
        f"fail_frames: {fail_frames}",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print("Phase6 validation complete:", report_path)


if __name__ == "__main__":
    main()

