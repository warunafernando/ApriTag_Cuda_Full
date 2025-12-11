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
from tests.helpers import rms, save_numpy, write_csv

try:
    from gpu.sampling import sample_gpu
except Exception as exc:  # pragma: no cover - defensive import
    sample_gpu = None
    _gpu_import_error = exc
else:
    _gpu_import_error = None


def main() -> None:
    cfg = load_config()
    ensure_output_dirs(cfg)

    if sample_gpu is None:
        raise RuntimeError(
            f"GPU sampling module unavailable: {_gpu_import_error}"
        )

    video_path = cfg["input"]["video_path"]
    frames = load_frames(video_path, range(5))

    debug_dir = Path("outputs/debug_phase1")
    metrics_csv = Path("outputs/phase1_sampling_smoke5.csv")

    rows = []
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

        save_numpy(debug_dir / f"sample_cpu_frame{idx:03d}.npy", cpu_sample)
        save_numpy(debug_dir / f"sample_gpu_frame{idx:03d}.npy", gpu_sample)

        rms_val = rms(cpu_sample, gpu_sample)
        rows.append([idx, rms_val, float(np.max(np.abs(cpu_sample - gpu_sample)))])
        (debug_dir / f"sample_diff_frame{idx:03d}.txt").write_text(
            f"RMS={rms_val}\nmax_error={float(np.max(np.abs(cpu_sample - gpu_sample)))}",
            encoding="utf-8",
        )

    write_csv(metrics_csv, ["frame", "rms", "max_error"], rows)
    print("GPU sampling smoke test complete")


if __name__ == "__main__":
    main()

