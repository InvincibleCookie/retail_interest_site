from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.config import OUTPUT_DIR
from app.services.storage import new_job_id, save_uploads, job_output_dir
from app.services.video import video_id, get_video_meta, save_frame_jpeg
from app.services.processor import analyze_videos
from app.services.feature_extraction import processing_device

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"

app = FastAPI(title="Retail Interest Zones", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

JOBS: dict[str, dict[str, Any]] = {}


@app.get("/")
def index():
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.post("/api/jobs")
async def create_job(files: list[UploadFile] = File(...)):
    job_id = new_job_id()
    videos = await save_uploads(job_id, files)
    if not videos:
        raise HTTPException(status_code=400, detail="Не найдено видео. Загрузите mp4/avi/mov/mkv или zip с видео.")

    items = []
    for p in videos:
        try:
            meta = get_video_meta(p)
        except Exception as e:
            meta = {"error": str(e)}
        items.append({
            "video_id": video_id(p),
            "filename": p.name,
            "path": str(p),
            "meta": meta,
        })

    JOBS[job_id] = {
        "job_id": job_id,
        "status": "uploaded",
        "progress": 0,
        "stage": "Ожидание запуска",
        "device": None,
        "elapsed_seconds": 0,
        "eta_seconds": None,
        "message": "Файлы загружены. Задайте ROI полки и запустите обработку.",
        "videos": [str(p) for p in videos],
        "items": items,
        "report": None,
        "error": None,
    }
    return {"job_id": job_id, "videos": items}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return JOBS[job_id]


@app.get("/api/jobs/{job_id}/videos/{vid}/frame")
def get_frame(job_id: str, vid: str, frame: int = 0):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    video_path = None
    for p in job["videos"]:
        if video_id(Path(p)) == vid:
            video_path = Path(p)
            break
    if video_path is None:
        raise HTTPException(status_code=404, detail="Video not found")

    out = job_output_dir(job_id) / "frames" / f"{vid}_{frame}.jpg"
    try:
        save_frame_jpeg(video_path, out, frame_idx=frame)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Не удалось получить кадр из видео: {exc}") from exc
    return FileResponse(
        out,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


def _run_processing(job_id: str, payload: dict[str, Any]):
    job = JOBS[job_id]
    started = time.monotonic()
    previous_eta: float | None = None
    job.update({
        "status": "processing",
        "progress": 1,
        "stage": "Инициализация",
        "device": processing_device(),
        "elapsed_seconds": 0,
        "eta_seconds": None,
        "_started_monotonic": started,
        "_eta_updated_elapsed": 0.0,
        "message": "Подготовка моделей и входных данных.",
        "error": None,
    })

    def update_progress(fraction: float, stage: str, message: str) -> None:
        nonlocal previous_eta
        fraction = max(0.0, min(1.0, float(fraction)))
        elapsed = max(0.0, time.monotonic() - started)
        raw_eta = elapsed * (1.0 - fraction) / fraction if fraction >= 0.01 else None
        if raw_eta is not None:
            previous_eta = raw_eta if previous_eta is None else previous_eta * 0.7 + raw_eta * 0.3
        job.update({
            "progress": min(99, max(1, round(fraction * 100))),
            "stage": stage,
            "message": message,
            "elapsed_seconds": round(elapsed),
            "eta_seconds": round(previous_eta) if previous_eta is not None else None,
            "_eta_updated_elapsed": elapsed,
        })

    try:
        videos = [Path(p) for p in job["videos"]]
        rois = payload.get("rois", {}) or {}
        params = payload.get("params", {}) or {}
        result = analyze_videos(
            job_id,
            videos,
            rois,
            params,
            progress_callback=update_progress,
        )

        elapsed = round(time.monotonic() - started)
        job.update({
            "status": "done",
            "progress": 100,
            "stage": "Готово",
            "message": "Обработка завершена, отчет сформирован.",
            "elapsed_seconds": elapsed,
            "eta_seconds": 0,
            "report": result["report"],
        })
    except Exception as e:
        job.update({
            "status": "error",
            "progress": 100,
            "stage": "Ошибка",
            "message": "Ошибка обработки.",
            "elapsed_seconds": round(time.monotonic() - started),
            "eta_seconds": None,
            "error": str(e),
        })


@app.post("/api/jobs/{job_id}/process")
def process_job(job_id: str, payload: dict[str, Any] = Body(...)):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    if JOBS[job_id]["status"] == "processing":
        raise HTTPException(status_code=409, detail="Job already processing")

    thread = threading.Thread(target=_run_processing, args=(job_id, payload), daemon=True)
    thread.start()
    return {"job_id": job_id, "status": "processing"}


@app.get("/api/jobs/{job_id}/status")
def job_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    job = JOBS[job_id]
    elapsed_seconds = job.get("elapsed_seconds", 0)
    eta_seconds = job.get("eta_seconds")
    if job.get("status") == "processing" and job.get("_started_monotonic") is not None:
        elapsed_seconds = round(time.monotonic() - job["_started_monotonic"])
        if eta_seconds is not None:
            since_estimate = elapsed_seconds - job.get("_eta_updated_elapsed", elapsed_seconds)
            eta_seconds = max(0, round(eta_seconds - since_estimate))
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job.get("progress", 0),
        "stage": job.get("stage", ""),
        "device": job.get("device"),
        "elapsed_seconds": elapsed_seconds,
        "eta_seconds": eta_seconds,
        "message": job.get("message", ""),
        "error": job.get("error"),
        "report": job.get("report"),
    }


@app.get("/api/jobs/{job_id}/report")
def get_report(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("status") != "done":
        raise HTTPException(status_code=409, detail="Report is not ready")
    return job.get("report")


@app.get("/api/jobs/{job_id}/download")
def download_report(job_id: str):
    job = JOBS.get(job_id)
    if not job or not job.get("report"):
        raise HTTPException(status_code=404, detail="Report not found")
    zip_path = Path(job["report"]["zip_report"])
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="Zip report not found")
    return FileResponse(zip_path, filename=zip_path.name, media_type="application/zip")
