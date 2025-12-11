"""
GPU corner (quad) extraction using CuPy for phase 4.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import cv2

try:
    import cupy as cp
except Exception as exc:  # pragma: no cover
    cp = None
    _gpu_import_error = exc
else:
    _gpu_import_error = None

from common.geometry import canonicalize_corners


def _aruco_params_from_cfg(cfg: Dict[str, Any]):
    params_cfg = cfg.get("aruco_params") or {
        # Adopted best profile from Phase-13 sweep: fast2
        "adaptiveThreshWinSizeMin": 3,
        "adaptiveThreshWinSizeMax": 11,
        "adaptiveThreshWinSizeStep": 4,
        "minMarkerPerimeterRate": 0.07,
        "maxMarkerPerimeterRate": 3.0,
        "cornerRefinementWinSize": 2,
        "cornerRefinementMaxIterations": 15,
    }
    aruco = cv2.aruco
    if hasattr(aruco, "DetectorParameters_create"):
        params = aruco.DetectorParameters_create()
    else:
        params = aruco.DetectorParameters()
    for k, v in params_cfg.items():
        if hasattr(params, k):
            setattr(params, k, v)
    return params


def detect_quad_gpu(gray: np.ndarray, cfg: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    GPU quad detector returning 4x2 corners or None.
    Optimized to rely on ArUco detection only (fast path, no extra passes).
    """
    # ArUco detection (CPU, fast for single tag)
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    params = _aruco_params_from_cfg(cfg)
    if hasattr(aruco, "DetectorParameters_create"):
        corners_list, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=params)
    else:
        detector = aruco.ArucoDetector(dictionary, params)
        corners_list, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return None

    ids_flat = ids.flatten()
    expected_id = cfg.get("tag", {}).get("id_expected")
    idx = 0
    if expected_id is not None:
        matches = np.where(ids_flat == expected_id)[0]
        if len(matches) > 0:
            idx = int(matches[0])

    quad = corners_list[idx].reshape(4, 2).astype(np.float32)
    return canonicalize_corners(quad)

