from __future__ import annotations

from pathlib import Path
import cv2


def video_id(path: Path) -> str:
    stem = Path(path).stem
    for suffix in ["_crop", "_video", "_processed"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def get_video_meta(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    meta = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return meta


def save_frame_jpeg(video_path: Path, out_path: Path, frame_idx: int = 0) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        requested_idx = max(0, int(frame_idx))
        if n_frames > 0:
            requested_idx = min(requested_idx, n_frames - 1)

        attempts = [requested_idx]
        if requested_idx != 0:
            attempts.append(0)
        attempts.extend(range(1, min(max(n_frames, 30), 30)))

        frame = None
        used_idx = requested_idx
        for candidate_idx in dict.fromkeys(attempts):
            cap.set(cv2.CAP_PROP_POS_FRAMES, candidate_idx)
            ok, candidate = cap.read()
            if ok and candidate is not None and candidate.size > 0:
                frame = candidate
                used_idx = candidate_idx
                break

        if frame is None:
            raise RuntimeError(f"Cannot read preview frame from video: {video_path}")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
            raise RuntimeError(f"Cannot save preview frame {used_idx}: {out_path}")
        return out_path
    finally:
        cap.release()
