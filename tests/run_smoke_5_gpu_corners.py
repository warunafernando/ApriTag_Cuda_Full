from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import ensure_output_dirs, load_config
from common.geometry import canonicalize_corners
from common.video import load_frames
from cpu.detector import detect_apriltag
from gpu.corners import detect_quad_gpu
from tests.helpers import save_overlay, write_csv


def metrics(cpu_corners: np.ndarray, gpu_corners: np.ndarray) -> tuple[float, float, float]:
    rms = float(np.sqrt(np.mean((cpu_corners - gpu_corners) ** 2)))
    cpu_centroid = cpu_corners.mean(axis=0)
    gpu_centroid = gpu_corners.mean(axis=0)
    centroid_delta = float(np.linalg.norm(cpu_centroid - gpu_centroid))
    cpu_area = _quad_area(cpu_corners)
    gpu_area = _quad_area(gpu_corners)
    area_ratio = float(gpu_area / cpu_area) if cpu_area > 0 else np.nan
    return rms, centroid_delta, area_ratio


def _quad_area(corners: np.ndarray) -> float:
    x = corners[:, 0]
    y = corners[:, 1]
    return 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def main() -> None:
    cfg = load_config()
    ensure_output_dirs(cfg)

    frames = load_frames(cfg["input"]["video_path"], range(5))
    csv_path = Path("outputs/phase4_smoke5_corners.csv")
    debug_dir = Path("outputs/debug_phase4")

    rows = []
    for idx, frame in enumerate(frames):
        detection = detect_apriltag(frame, cfg)
        if not detection.detected or detection.corners is None:
            rows.append([idx, False, None, None, None, None])
            continue

        cpu_corners = detection.corners
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gpu_corners = detect_quad_gpu(gray, cfg)

        if gpu_corners is None:
            rows.append([idx, True, False, None, None, None])
            continue

        rms, centroid_delta, area_ratio = metrics(cpu_corners, gpu_corners)
        rows.append([idx, True, True, rms, centroid_delta, area_ratio])

        # overlays
        save_overlay(frame, cpu_corners, detection.id, debug_dir / f"frame_{idx:03d}_cpu_overlay.png")
        save_overlay(frame, gpu_corners, detection.id, debug_dir / f"frame_{idx:03d}_gpu_overlay.png")

    write_csv(
        csv_path,
        ["frame", "cpu_detected", "gpu_detected", "rms", "centroid_delta", "area_ratio"],
        rows,
    )
    mean_rms = np.nanmean([r[3] for r in rows if r[3] is not None])
    print(f"Smoke GPU corners complete. Mean RMS: {mean_rms}")


if __name__ == "__main__":
    main()

