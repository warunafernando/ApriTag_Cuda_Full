from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import ensure_output_dirs, load_config
from common.video import load_frames
from cpu.pipeline import run_cpu_pipeline
from tests.helpers import write_csv


def main() -> None:
    cfg = load_config()
    ensure_output_dirs(cfg)

    video_path = cfg["input"]["video_path"]
    frames = load_frames(video_path, range(200))

    metrics_csv = Path("outputs/phase0_validate200_cpu.csv")
    report_txt = Path("outputs/phase0_report.txt")

    rows = []
    detections = 0

    for idx, frame in enumerate(frames):
        detection, decode = run_cpu_pipeline(frame, cfg)
        if detection.detected and detection.id == cfg["tag"]["id_expected"]:
            detections += 1

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
    report_lines = [
        f"frames_processed: {len(frames)}",
        f"detection_rate: {detection_rate:.4f}",
    ]
    report_txt.parent.mkdir(parents=True, exist_ok=True)
    report_txt.write_text("\n".join(report_lines), encoding="utf-8")
    print("Validation complete. Detection rate:", detection_rate)


if __name__ == "__main__":
    main()

