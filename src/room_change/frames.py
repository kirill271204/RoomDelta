"""Pick the N sharpest, evenly spaced frames from a video.

The video is split into N equal time windows and the frame with the highest
Laplacian variance (a standard sharpness score) is kept from each window, so
motion blurred frames are skipped while coverage of the whole clip is kept.
"""

import re
from pathlib import Path

import cv2
import numpy as np


def video_name(path: Path) -> str:
    """Folder-safe name for a video, e.g. 'IMG_1765 2.MOV' -> 'IMG_1765_2'."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", Path(path).stem).strip("_")


def sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def extract_frames(video: Path, out_dir: Path, n: int, long_side: int = 1024) -> list[Path]:
    cap = cv2.VideoCapture(str(video))  # OpenCV applies the iPhone rotation tag
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        raise SystemExit(f"Could not read frames from {video}")

    bounds = np.linspace(0, total, n + 1).astype(int)
    best = [(-1.0, None) for _ in range(n)]  # (score, frame) per window
    for idx in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        w = min(np.searchsorted(bounds, idx, side="right") - 1, n - 1)
        # Score on a downscaled copy: same ranking, much faster on 4K input
        score = sharpness(cv2.resize(frame, None, fx=0.25, fy=0.25))
        if score > best[w][0]:
            best[w] = (score, frame)
    cap.release()

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i, (_, frame) in enumerate(best):
        if frame is None:
            continue
        h, w = frame.shape[:2]
        scale = long_side / max(h, w)
        if scale < 1:
            frame = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        path = out_dir / f"{i:03d}.jpg"
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        written.append(path)
    return written
