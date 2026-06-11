from __future__ import annotations

import asyncio
import sys
import types
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pytest
from fastapi import HTTPException, UploadFile

# The API tests monkeypatch video operations and do not need real OpenCV.
# Stubbing cv2 keeps the tests runnable in headless environments where an
# already-installed opencv-python wheel may require system libGL.
sys.modules.setdefault("cv2", types.SimpleNamespace())

import app.main as main


class ImmediateThread:
    def __init__(self, target, args=(), daemon=None):
        self.target = target
        self.args = args
        self.daemon = daemon

    def start(self) -> None:
        self.target(*self.args)


def _upload(name: str = "sample.mp4") -> UploadFile:
    return UploadFile(filename=name, file=BytesIO(b"video"))


def _zip_report(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w") as archive:
        archive.writestr("report/overview.csv", "metric,value\nvideos_total,1\n")
    return path


def test_api_job_lifecycle_frame_process_report_and_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    main.JOBS.clear()
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"video")
    zip_path = _zip_report(tmp_path / "retail_interest_report_job-api.zip")

    async def fake_save_uploads(job_id, files):
        return [video_path]

    def fake_analyze_videos(job_id, videos, rois, params, progress_callback=None):
        if progress_callback:
            progress_callback(0.5, "Классификация действий", "Проверка прогресса")
            progress_callback(1.0, "Готово", "Отчет сформирован")
        return {
            "report": {
                "overview": [{"metric": "videos_total", "value": len(videos)}],
                "video_summary": [{"video_id": "sample", "n_events": 1, "n_zones": 1}],
                "zip_report": str(zip_path),
                "html_report_url": f"/outputs/{job_id}/report/retail_interest_report.html",
            }
        }

    def fake_save_frame(video_path_arg, out_path, frame_idx=0):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"jpeg")
        return out_path

    monkeypatch.setattr(main, "new_job_id", lambda: "job-api")
    monkeypatch.setattr(main, "save_uploads", fake_save_uploads)
    monkeypatch.setattr(main, "get_video_meta", lambda path: {"width": 640, "height": 480, "fps": 25.0, "n_frames": 100})
    monkeypatch.setattr(main, "save_frame_jpeg", fake_save_frame)
    monkeypatch.setattr(main, "analyze_videos", fake_analyze_videos)
    monkeypatch.setattr(main.threading, "Thread", ImmediateThread)

    created = asyncio.run(main.create_job([_upload()]))
    assert created["job_id"] == "job-api"
    assert created["videos"][0]["video_id"] == "sample"
    assert created["videos"][0]["meta"]["width"] == 640

    job = main.get_job("job-api")
    assert job["status"] == "uploaded"
    assert len(job["items"]) == 1

    status_before = main.job_status("job-api")
    assert status_before["status"] == "uploaded"
    assert status_before["progress"] == 0

    frame = main.get_frame("job-api", "sample", frame=0)
    assert frame.media_type == "image/jpeg"
    assert Path(frame.path).read_bytes() == b"jpeg"

    with pytest.raises(HTTPException) as report_not_ready:
        main.get_report("job-api")
    assert report_not_ready.value.status_code == 409

    process = main.process_job("job-api", {"rois": {"sample": [[0, 0], [1, 0], [1, 1]]}, "params": {}})
    assert process == {"job_id": "job-api", "status": "processing"}

    status_done = main.job_status("job-api")
    assert status_done["status"] == "done"
    assert status_done["progress"] == 100
    assert status_done["stage"] == "Готово"
    assert status_done["device"] in {"CPU", "CUDA: Test GPU"} or status_done["device"].startswith("CUDA:")
    assert status_done["elapsed_seconds"] >= 0
    assert status_done["eta_seconds"] == 0
    assert status_done["report"]["overview"][0] == {"metric": "videos_total", "value": 1}

    report = main.get_report("job-api")
    assert report["zip_report"] == str(zip_path)

    download = main.download_report("job-api")
    assert download.media_type == "application/zip"
    assert Path(download.path) == zip_path


def test_api_error_statuses_for_missing_and_busy_jobs() -> None:
    main.JOBS.clear()

    for call in [
        lambda: main.get_job("missing"),
        lambda: main.job_status("missing"),
        lambda: main.process_job("missing", {"rois": {}}),
        lambda: main.download_report("missing"),
    ]:
        with pytest.raises(HTTPException) as exc_info:
            call()
        assert exc_info.value.status_code == 404

    main.JOBS["busy"] = {"job_id": "busy", "status": "processing"}
    with pytest.raises(HTTPException) as busy_exc:
        main.process_job("busy", {"rois": {}})
    assert busy_exc.value.status_code == 409


def test_frame_api_returns_422_when_video_preview_cannot_be_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    main.JOBS.clear()
    video_path = tmp_path / "broken.mp4"
    video_path.write_bytes(b"broken")
    main.JOBS["broken-job"] = {
        "job_id": "broken-job",
        "status": "uploaded",
        "videos": [str(video_path)],
    }
    monkeypatch.setattr(main, "save_frame_jpeg", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("decode failed")))

    with pytest.raises(HTTPException) as exc_info:
        main.get_frame("broken-job", "broken", frame=0)

    assert exc_info.value.status_code == 422
    assert "Не удалось получить кадр" in str(exc_info.value.detail)
