from __future__ import annotations

import shutil
import unicodedata
import zipfile
import uuid
from pathlib import Path
from typing import Iterable

from fastapi import UploadFile

from app.config import UPLOAD_DIR, OUTPUT_DIR, ALLOWED_VIDEO_EXT, ALLOWED_ARCHIVE_EXT


def new_job_id() -> str:
    return uuid.uuid4().hex[:12]


def safe_name(name: str) -> str:
    name = Path(name).name
    keep = []
    for ch in name:
        if ch.isalnum() or ch in {".", "_", "-", " "}:
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip().replace(" ", "_")
    return out or "file"


def storage_name(name: str, default_stem: str = "video") -> str:
    safe = safe_name(name)
    path = Path(safe)
    suffix = path.suffix.lower()
    normalized = unicodedata.normalize("NFKD", path.stem)
    ascii_stem = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_stem = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in ascii_stem)
    ascii_stem = ascii_stem.strip("_.-")
    if not ascii_stem:
        ascii_stem = f"{default_stem}_{uuid.uuid4().hex[:8]}"
    return f"{ascii_stem}{suffix}"


def unique_path(directory: Path, filename: str) -> Path:
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    return directory / f"{stem}_{uuid.uuid4().hex[:8]}{suffix}"


def job_upload_dir(job_id: str) -> Path:
    p = UPLOAD_DIR / job_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def job_output_dir(job_id: str) -> Path:
    p = OUTPUT_DIR / job_id
    p.mkdir(parents=True, exist_ok=True)
    return p


async def save_uploads(job_id: str, files: Iterable[UploadFile]) -> list[Path]:
    base = job_upload_dir(job_id)
    saved: list[Path] = []

    for file in files:
        original_name = safe_name(file.filename or "upload")
        suffix = Path(original_name).suffix.lower()
        filename = storage_name(original_name, default_stem="upload")
        dst = unique_path(base, filename)

        with open(dst, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)

        if suffix in ALLOWED_ARCHIVE_EXT:
            extract_dir = base / f"{Path(filename).stem}_extracted"
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(dst, "r") as zf:
                for member in zf.infolist():
                    if member.is_dir():
                        continue
                    original_member_name = safe_name(Path(member.filename).name)
                    if Path(original_member_name).suffix.lower() not in ALLOWED_VIDEO_EXT:
                        continue
                    member_name = storage_name(original_member_name, default_stem="video")
                    target = unique_path(extract_dir, member_name)
                    with zf.open(member) as src, open(target, "wb") as out:
                        shutil.copyfileobj(src, out)
                    saved.append(target)
        elif suffix in ALLOWED_VIDEO_EXT:
            saved.append(dst)

    return saved
