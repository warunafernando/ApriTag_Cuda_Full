from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from common.config import ensure_output_dirs, load_config
from common.video import load_frames
from cpu.detector import detect_apriltag
from cpu.decode import decode_apriltag
from tests.helpers import rms, write_csv

try:
    from gpu.sampling import sample_gpu
except Exception as exc:  # pragma: no cover
    sample_gpu = None
    _gpu_import_error = exc
else:
    _gpu_import_error = None


def main() -> None:
    cfg = load_config()
    ensure_output_dirs(cfg)

    if sample_gpu is None:
        raise RuntimeError(f"GPU sampling module unavailable: {_gpu_import_error}")

    video_path = cfg["input"]["video_path"]
    frames = load_frames(video_path, range(200))

    report_path = Path("outputs/phase1_sampling_report.txt")
    rows = []
    rms_values = []

    for idx, frame in enumerate(frames):
        detection = detect_apriltag(frame, cfg)
        if not detection.detected or detection.corners is None:
            rows.append([idx, None, None])
            continue

        gray = frame
        if frame.ndim == 3:
            import cv2
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cpu_decode = decode_apriltag(gray, detection.corners, cfg)
        cpu_sample = cpu_decode.sample_grid
        gpu_sample = sample_gpu(gray, detection.corners, cfg)

        rms_val = rms(cpu_sample, gpu_sample)
        max_err = float(np.max(np.abs(cpu_sample - gpu_sample)))
        rows.append([idx, rms_val, max_err])
        rms_values.append(rms_val)

    write_csv(Path("outputs/phase1_sampling_200.csv"), ["frame", "rms", "max_error"], rows)

    mean_rms = float(np.mean(rms_values)) if rms_values else float("nan")
    max_rms = float(np.max(rms_values)) if rms_values else float("nan")
    report_lines = [
        f"frames_processed: {len(frames)}",
        f"mean_rms: {mean_rms}",
        f"max_rms: {max_rms}",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print("GPU sampling validation complete")


if __name__ == "__main__":
    main()

