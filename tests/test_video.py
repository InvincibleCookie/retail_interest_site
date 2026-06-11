from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

cv2_stub = types.SimpleNamespace(
    CAP_PROP_FRAME_COUNT=7,
    CAP_PROP_POS_FRAMES=1,
    IMWRITE_JPEG_QUALITY=95,
)
sys.modules.setdefault("cv2", cv2_stub)

from app.services import video


class FakeCapture:
    def __init__(self, frames: dict[int, np.ndarray | None], n_frames: int = 10):
        self.frames = frames
        self.n_frames = n_frames
        self.position = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def get(self, prop):
        if prop == video.cv2.CAP_PROP_FRAME_COUNT:
            return self.n_frames
        return 0

    def set(self, prop, value):
        self.position = int(value)
        return True

    def read(self):
        frame = self.frames.get(self.position)
        return frame is not None, frame

    def release(self):
        self.released = True


def test_save_frame_jpeg_falls_back_to_next_readable_frame(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(video.cv2, "CAP_PROP_FRAME_COUNT", 7, raising=False)
    monkeypatch.setattr(video.cv2, "CAP_PROP_POS_FRAMES", 1, raising=False)
    monkeypatch.setattr(video.cv2, "IMWRITE_JPEG_QUALITY", 95, raising=False)
    frame = np.ones((8, 8, 3), dtype=np.uint8)
    capture = FakeCapture({0: None, 1: frame})
    written = {}

    monkeypatch.setattr(video.cv2, "VideoCapture", lambda path: capture, raising=False)
    monkeypatch.setattr(video.cv2, "imwrite", lambda path, image, params: written.update(path=path, image=image) or True, raising=False)

    out = video.save_frame_jpeg(tmp_path / "input.mp4", tmp_path / "preview.jpg", frame_idx=0)

    assert out == tmp_path / "preview.jpg"
    assert written["path"] == str(out)
    assert np.array_equal(written["image"], frame)
    assert capture.released is True


def test_save_frame_jpeg_raises_when_no_frame_can_be_decoded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(video.cv2, "CAP_PROP_FRAME_COUNT", 7, raising=False)
    monkeypatch.setattr(video.cv2, "CAP_PROP_POS_FRAMES", 1, raising=False)
    capture = FakeCapture({}, n_frames=3)
    monkeypatch.setattr(video.cv2, "VideoCapture", lambda path: capture, raising=False)

    with pytest.raises(RuntimeError, match="Cannot read preview frame"):
        video.save_frame_jpeg(tmp_path / "broken.mp4", tmp_path / "preview.jpg")

    assert capture.released is True
