from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("RETAIL_WEB_DATA_DIR", BASE_DIR / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
JOB_DIR = DATA_DIR / "jobs"

for p in [DATA_DIR, UPLOAD_DIR, OUTPUT_DIR, JOB_DIR]:
    p.mkdir(parents=True, exist_ok=True)

ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
ALLOWED_ARCHIVE_EXT = {".zip"}
