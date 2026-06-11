from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import asyncio

import pytest
from fastapi import UploadFile

from app.services import storage


def _upload_file(name: str, content: bytes) -> UploadFile:
    return UploadFile(filename=name, file=BytesIO(content))


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def test_safe_name_strips_directories_and_replaces_unsafe_characters() -> None:
    assert storage.safe_name("../nested/магазин demo?.mp4") == "магазин_demo_.mp4"
    assert storage.safe_name("***") == "___"
    assert storage.safe_name("   ") == "file"


def test_save_uploads_accepts_supported_video_and_ignores_other_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(storage, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(storage, "OUTPUT_DIR", tmp_path / "outputs")

    saved = asyncio.run(storage.save_uploads(
        "job-video",
        [
            _upload_file("camera 1.mp4", b"video-bytes"),
            _upload_file("notes.txt", b"not a video"),
        ],
    ))

    assert [path.name for path in saved] == ["camera_1.mp4"]
    assert saved[0].read_bytes() == b"video-bytes"
    assert saved[0].is_relative_to(tmp_path / "uploads" / "job-video")


def test_save_uploads_extracts_only_videos_from_zip_with_sanitized_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(storage, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(storage, "OUTPUT_DIR", tmp_path / "outputs")
    archive = _zip_bytes(
        {
            "nested/camera one.mp4": b"mp4-bytes",
            "../escape.mkv": b"mkv-bytes",
            "nested/readme.txt": b"ignored",
        }
    )

    saved = asyncio.run(storage.save_uploads("job-zip", [_upload_file("dataset.zip", archive)]))

    assert [path.name for path in saved] == ["camera_one.mp4", "escape.mkv"]
    assert [path.read_bytes() for path in saved] == [b"mp4-bytes", b"mkv-bytes"]
    for path in saved:
        assert path.is_relative_to(tmp_path / "uploads" / "job-zip" / "dataset_extracted")


def test_storage_name_converts_unicode_video_name_to_ascii() -> None:
    filename = storage.storage_name("Видео магазина.mp4")

    assert filename.endswith(".mp4")
    assert filename.isascii()
    assert filename.startswith("video_")


def test_save_uploads_uses_ascii_path_for_unicode_video_on_windows_compatible_flow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(storage, "UPLOAD_DIR", tmp_path / "uploads")

    saved = asyncio.run(
        storage.save_uploads(
            "job-unicode",
            [_upload_file("Видео.mp4", b"video-bytes")],
        )
    )

    assert len(saved) == 1
    assert saved[0].name.isascii()
    assert saved[0].suffix == ".mp4"
    assert saved[0].read_bytes() == b"video-bytes"
