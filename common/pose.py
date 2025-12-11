from __future__ import annotations

from typing import Any, Dict, Tuple

import cv2
import numpy as np


def camera_matrix_from_image(shape: Tuple[int, int], fx: float | None = None, fy: float | None = None):
    h, w = shape
    if fx is None or fy is None:
        focal = max(h, w)
        fx = fy = float(focal)
    cx = w / 2.0
    cy = h / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)
    return K, dist


def solve_pose_cpu(corners: np.ndarray, image_shape: Tuple[int, int], cfg: Dict[str, Any]):
    """
    Solve pose using CPU (solvePnP) given ordered corners TL,TR,BR,BL.
    """
    tag_size = float(cfg.get("pose", {}).get("tag_size_meters", 0.16))
    use_ippe = bool(cfg.get("pose", {}).get("use_ippe", True))

    objp = np.array(
        [
            [-tag_size / 2, tag_size / 2, 0],
            [tag_size / 2, tag_size / 2, 0],
            [tag_size / 2, -tag_size / 2, 0],
            [-tag_size / 2, -tag_size / 2, 0],
        ],
        dtype=np.float32,
    )
    imgp = corners.astype(np.float32)

    K, dist = camera_matrix_from_image(image_shape)

    flags = cv2.SOLVEPNP_IPPE_SQUARE if use_ippe else cv2.SOLVEPNP_ITERATIVE
    success, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist, flags=flags)
    if not success and flags != cv2.SOLVEPNP_ITERATIVE:
        success, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return None, None
    return rvec.astype(np.float64), tvec.astype(np.float64)


def solve_pose_gpu_from_homography(corners: np.ndarray, image_shape: Tuple[int, int], cfg: Dict[str, Any]):
    """
    Solve pose via homography decomposition (CPU math, GPU flag only).
    """
    if cfg.get("pose", {}).get("gpu", {}).get("force_cpu_pose", False):
        return solve_pose_cpu(corners, image_shape, cfg)

    tag_size = float(cfg.get("pose", {}).get("tag_size_meters", 0.16))

    objp = np.array(
        [
            [-tag_size / 2, tag_size / 2],
            [tag_size / 2, tag_size / 2],
            [tag_size / 2, -tag_size / 2],
            [-tag_size / 2, -tag_size / 2],
        ],
        dtype=np.float32,
    )
    imgp = corners.astype(np.float32)

    H, _ = cv2.findHomography(objp, imgp, method=0)
    if H is None:
        return solve_pose_cpu(corners, image_shape, cfg)

    K, _ = camera_matrix_from_image(image_shape)
    K_inv = np.linalg.inv(K)
    h1 = H[:, 0]
    h2 = H[:, 1]
    h3 = H[:, 2]

    Kinv_h1 = K_inv @ h1
    Kinv_h2 = K_inv @ h2
    Kinv_h3 = K_inv @ h3

    norm1 = np.linalg.norm(Kinv_h1)
    norm2 = np.linalg.norm(Kinv_h2)
    if norm1 <= 1e-9 or norm2 <= 1e-9:
        return solve_pose_cpu(corners, image_shape, cfg)

    lam = (norm1 + norm2) / 2.0
    r1 = Kinv_h1 / norm1
    r2 = Kinv_h2 / norm2
    r3 = np.cross(r1, r2)

    R = np.stack([r1, r2, r3], axis=1)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1

    t = Kinv_h3 / lam
    rvec, _ = cv2.Rodrigues(R)
    rvec = rvec.astype(np.float64)
    t = t.reshape(3, 1).astype(np.float64)

    # Optional refinement using solvePnP with the homography pose as an initial guess.
    objp3d = np.array(
        [
            [-tag_size / 2, tag_size / 2, 0],
            [tag_size / 2, tag_size / 2, 0],
            [tag_size / 2, -tag_size / 2, 0],
            [-tag_size / 2, -tag_size / 2, 0],
        ],
        dtype=np.float32,
    )
    dist = np.zeros(5, dtype=np.float64)
    try:
        success, rvec_ref, tvec_ref = cv2.solvePnP(
            objp3d,
            imgp,
            K,
            dist,
            rvec,
            t,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if success:
            rvec, t = rvec_ref.astype(np.float64), tvec_ref.astype(np.float64)
    except cv2.error:
        pass

    return rvec, t


def rotation_error_deg(R_ref: np.ndarray, R_test: np.ndarray) -> float:
    R_rel = R_test @ R_ref.T
    angle = np.arccos(np.clip((np.trace(R_rel) - 1) / 2.0, -1.0, 1.0))
    return float(np.degrees(angle))


def translation_error(t_ref: np.ndarray, t_test: np.ndarray) -> float:
    return float(np.linalg.norm(t_ref.reshape(3) - t_test.reshape(3)))

