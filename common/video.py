from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np


def load_frames(video_path: str | Path, frame_indices: Iterable[int]) -> List[np.ndarray]:
    """
    Load specific frames from a video file.

    Args:
        video_path: Path to the video file.
        frame_indices: Iterable of zero-based frame indices to fetch.

    Returns:
        List of frames as BGR numpy arrays in the same order as requested.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {idx} from {path}")
        frames.append(frame)

    cap.release()
    return frames


def load_frame_range(video_path: str | Path, start: int, end_inclusive: int) -> List[np.ndarray]:
    """
    Convenience helper to load a contiguous range of frames.
    """
    return load_frames(video_path, range(start, end_inclusive + 1))

