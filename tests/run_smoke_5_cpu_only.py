from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import ensure_output_dirs, load_config
from common.video import load_frames
from cpu.pipeline import run_cpu_pipeline
from tests.helpers import save_overlay, write_csv


def main() -> None:
    cfg = load_config()
    ensure_output_dirs(cfg)

    video_path = cfg["input"]["video_path"]
    frames = load_frames(video_path, range(5))

    debug_dir = Path("outputs/debug_phase0")
    metrics_csv = Path("outputs/phase0_smoke5_cpu.csv")

    rows = []
    detections = 0

    for idx, frame in enumerate(frames):
        detection, decode = run_cpu_pipeline(frame, cfg)
        if detection.detected and detection.id == cfg["tag"]["id_expected"]:
            detections += 1

        save_overlay(
            frame,
            detection.corners if detection.detected else None,
            detection.id if detection.detected else None,
            debug_dir / f"frame_{idx:03d}_cpu_overlay.png",
        )

        rows.append(
            [
                idx,
                detection.detected,
                detection.id,
                decode.id,
                decode.hamming,
            ]
        )

    write_csv(
        metrics_csv,
        ["frame", "detected", "id_detected", "id_decoded", "hamming"],
        rows,
    )

    detection_rate = detections / len(frames)
    print(f"Smoke 5 CPU-only detection rate: {detection_rate:.2f}")


if __name__ == "__main__":
    main()

