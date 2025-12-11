from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from common.geometry import canonicalize_corners
from common.pose import rotation_error_deg, translation_error


DEFAULT_LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


@dataclass
class TagTrackerState:
    has_valid: bool = False
    corners: np.ndarray | None = None  # (4,2)
    pose_rvec: np.ndarray | None = None  # (3,1)
    pose_tvec: np.ndarray | None = None  # (3,1)
    last_frame_idx: int = -1


def should_use_detect_mode(frame_idx: int, tracker: TagTrackerState, tracking_cfg: dict) -> bool:
    if not tracking_cfg.get("enabled", True):
        return True
    if not tracker.has_valid:
        return True
    detect_every_n = int(tracking_cfg.get("detect_every_n", 0))
    if detect_every_n > 0 and frame_idx % detect_every_n == 0:
        return True
    return False


def track_corners_klt(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    prev_corners: np.ndarray,
    max_err: float,
    min_points: int,
) -> tuple[np.ndarray | None, bool]:
    prev_pts = prev_corners.reshape(-1, 1, 2).astype(np.float32)
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **DEFAULT_LK_PARAMS)
    status = status.reshape(-1)
    err = err.reshape(-1)
    good_mask = (status == 1) & (err <= max_err)
    if good_mask.sum() < min_points:
        return None, False
    tracked = curr_pts.reshape(-1, 2)
    tracked = canonicalize_corners(tracked)
    return tracked, True


def pose_delta(cpu_rvec_prev: np.ndarray, cpu_tvec_prev: np.ndarray, cpu_rvec: np.ndarray, cpu_tvec: np.ndarray) -> Tuple[float, float]:
    R_prev, _ = cv2.Rodrigues(cpu_rvec_prev)
    R_cur, _ = cv2.Rodrigues(cpu_rvec)
    rot_err = rotation_error_deg(R_prev, R_cur)
    trans_err = translation_error(cpu_tvec_prev, cpu_tvec)
    return rot_err, trans_err

