from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import ensure_output_dirs, load_config
from common.video import load_frames
from cpu.detector import detect_apriltag
from cpu.decode import decode_apriltag
from tests.helpers import write_csv

try:
    from gpu.sampling import sample_gpu
    from gpu.decode import decode_gpu_bits, decode_gpu_codebook
except Exception as exc:  # pragma: no cover
    sample_gpu = None
    decode_gpu_bits = None
    decode_gpu_codebook = None
    _gpu_import_error = exc
else:
    _gpu_import_error = None


def main() -> None:
    cfg = load_config()
    ensure_output_dirs(cfg)

    if None in (sample_gpu, decode_gpu_bits, decode_gpu_codebook):
        raise RuntimeError(f"GPU decode modules unavailable: {_gpu_import_error}")

    video_path = cfg["input"]["video_path"]
    frames = load_frames(video_path, range(200))

    bits_csv = Path("outputs/phase2_bits_200.csv")
    rows = []

    for idx, frame in enumerate(frames):
        detection = detect_apriltag(frame, cfg)
        if not detection.detected or detection.corners is None:
            rows.append([idx, None, None, None, None, None, False])
            continue

        gray = frame
        if frame.ndim == 3:
            import cv2
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cpu_decode = decode_apriltag(gray, detection.corners, cfg)
        cpu_bits = cpu_decode.bits
        cpu_hamming = cpu_decode.hamming
        cpu_id = cpu_decode.id

        gpu_sample = sample_gpu(gray, detection.corners, cfg)
        gpu_bits = decode_gpu_bits(gpu_sample, cfg)
        gpu_id, gpu_hamming = decode_gpu_codebook(gpu_bits, cfg)

        hamming_bits_diff = None
        if cpu_bits is not None and gpu_bits is not None:
            import numpy as np
            cpu_arr = np.asarray(cpu_bits, dtype=np.uint8).ravel()
            gpu_arr = np.asarray(gpu_bits, dtype=np.uint8).ravel()
            hamming_bits_diff = int(np.count_nonzero(cpu_arr != gpu_arr))

        rows.append(
            [
                idx,
                cpu_id,
                gpu_id,
                cpu_hamming,
                gpu_hamming,
                hamming_bits_diff,
                True,
            ]
        )

    write_csv(
        bits_csv,
        ["frame", "cpu_id", "gpu_id", "cpu_hamming", "gpu_hamming", "hamming_bits_diff", "detected"],
        rows,
    )
    print("GPU decode (CPU corners) validation complete")


if __name__ == "__main__":
    main()

