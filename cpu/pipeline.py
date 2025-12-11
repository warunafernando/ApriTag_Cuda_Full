from __future__ import annotations

from typing import Any, Dict, Tuple

import cv2

from cpu.decode import DecodeResultCPU, decode_apriltag
from cpu.detector import DetectionResultCPU, detect_apriltag


def run_cpu_pipeline(frame, cfg: Dict[str, Any]) -> Tuple[DetectionResultCPU, DecodeResultCPU]:
    """
    Convenience wrapper: detect then decode on CPU.
    """
    detection = detect_apriltag(frame, cfg)
    if not detection.detected or detection.corners is None:
        return detection, DecodeResultCPU(id=None, hamming=None, bits=None, sample_grid=None)

    gray = frame
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    decode = decode_apriltag(gray, detection.corners, cfg)
    return detection, decode

