from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import cv2
import numpy as np

from common.geometry import canonicalize_corners


@dataclass
class DetectionResultCPU:
    detected: bool
    id: Optional[int]
    corners: Optional[np.ndarray]  # (4,2) TL, TR, BR, BL


def detect_apriltag(frame: np.ndarray, cfg: Dict[str, Any]) -> DetectionResultCPU:
    """
    CPU reference detection using OpenCV ArUco AprilTag 36h11.
    """
    gray = frame
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)

    # OpenCV 4.10 switched to ArucoDetector; keep compatibility with older API.
    if hasattr(aruco, "DetectorParameters_create"):
        params = aruco.DetectorParameters_create()
        corners_list, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=params)
    else:
        params = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(dictionary, params)
        corners_list, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return DetectionResultCPU(False, None, None)

    ids_flat = ids.flatten()
    expected_id = cfg.get("tag", {}).get("id_expected")

    # Prefer expected ID when present; otherwise pick the first detection.
    idx = 0
    if expected_id is not None:
        matches = np.where(ids_flat == expected_id)[0]
        if len(matches) > 0:
            idx = int(matches[0])

    corners = corners_list[idx].reshape(-1, 2).astype(np.float32)

    # Refine to subpixel accuracy.
    term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    cv2.cornerSubPix(
        gray,
        corners,
        winSize=(3, 3),
        zeroZone=(-1, -1),
        criteria=term_crit,
    )

    ordered_corners = canonicalize_corners(corners)
    return DetectionResultCPU(True, int(ids_flat[idx]), ordered_corners)

