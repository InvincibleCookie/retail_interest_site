from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import time
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from fastapi import HTTPException, UploadFile

ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

main_module: Any | None = None
build_job_report: Callable[..., dict[str, Any]] | None = None
job_output_dir: Callable[[str], Path] | None = None
get_video_meta: Callable[[Path], dict[str, Any]] | None = None
video_id: Callable[[Path], str] | None = None

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
DEFAULT_ASSETS_DIR = ROOT_DIR / "service_test_assets"
DEFAULT_REPORT_DIR = ROOT_DIR / "data" / "test_reports"
POLL_TIMEOUT_SECONDS = 120
REAL_ML_TIMEOUT_SECONDS = 21600
POLL_INTERVAL_SECONDS = 0.5

ONE_PIXEL_JPEG = bytes.fromhex(
    "ffd8ffe000104a46494600010101006000600000ffdb0043000302020302020303030304030304050805050404050a070706"
    "080c0a0c0c0b0a0b0b0d0e12100d0e110e0b0b1016101113141515150c0f171816141812141514ffdb004301030404"
    "05040509050509140d0b0d141414141414141414141414141414141414141414141414141414141414141414141414141414"
    "1414141414141414141414141414141414141414ffc00011080001000103012200021101031101ffc400140001000000000000"
    "0000000000000000000008ffc4001410010000000000000000000000000000000000ffda000c03010002110311003f00b2c001ffd9"
)


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


class ServiceTestRunner:
    def __init__(
        self,
        assets_dir: Path,
        real_ml: bool,
        verbose_ml: bool,
        report_dir: Path,
        timeout_seconds: int | None = None,
    ) -> None:
        self.assets_dir = assets_dir
        self.real_ml = real_ml
        self.verbose_ml = verbose_ml
        self.report_dir = report_dir
        self.timeout_seconds = timeout_seconds or (REAL_ML_TIMEOUT_SECONDS if real_ml else POLL_TIMEOUT_SECONDS)
        self.results: list[CheckResult] = []
        self._last_processing_log = ""
        self._original_analyze: Callable[..., Any] | None = None

    def run(self) -> int:
        try:
            video = self._find_first_file(self.assets_dir / "videos", VIDEO_EXTENSIONS)
            archive = self._find_first_file(self.assets_dir / "archives", {".zip"})

            if video is None or archive is None:
                self._print_missing_assets(video, archive)
                return 2

            self._load_app_modules()
            self._install_fake_analyzer_if_needed()
            self._run_frontend_button_contract_checks()
            self._run_positive_case("Одиночное видео", video)
            self._run_positive_case("ZIP-архив с видео", archive)
            self._run_invalid_file_case()
            self._run_zip_without_video_case()
            self._run_missing_job_api_cases()
            self._print_summary()
            self._write_test_report()
            return 0 if all(item.ok for item in self.results) else 1
        finally:
            self._restore_analyzer()

    def _run_frontend_button_contract_checks(self) -> None:
        index_html = (BACKEND_DIR / "app" / "static" / "index.html").read_text(encoding="utf-8")
        app_js = (BACKEND_DIR / "app" / "static" / "app.js").read_text(encoding="utf-8")
        checks = [
            (
                "UI/API: кнопка загрузки",
                'id="uploadBtn"' in index_html and "fetch('/api/jobs'" in app_js and "method:'POST'" in app_js,
                "uploadBtn отправляет POST /api/jobs и ожидает job_id/videos",
            ),
            (
                "UI/API: кнопка запуска анализа",
                'id="processBtn"' in index_html and "/process`" in app_js and "method:'POST'" in app_js,
                "processBtn отправляет POST /api/jobs/{job_id}/process с ROI и params",
            ),
            (
                "UI/API: опрос статуса",
                "function pollStatus()" in app_js and "/status?t=" in app_js and "renderReport(data.report)" in app_js,
                "frontend опрашивает /status и рендерит отчет при статусе done",
            ),
            (
                "UI/API: кнопка скачивания отчета",
                'id="downloadLink"' in index_html and "downloadLink" in app_js and "/download" in app_js,
                "downloadLink получает href /api/jobs/{job_id}/download",
            ),
            (
                "UI/API: кнопка применить ROI ко всем",
                'id="applyAllBtn"' in index_html and "applyAllBtn" in app_js and "rois[v.video_id]" in app_js,
                "applyAllBtn копирует текущую ROI на все видео",
            ),
        ]
        for name, ok, detail in checks:
            self._record(name, ok, detail)

    def _load_app_modules(self) -> None:
        global build_job_report, get_video_meta, job_output_dir, main_module, video_id

        import app.main as loaded_main
        from app.services.reports import build_job_report as loaded_build_job_report
        from app.services.storage import job_output_dir as loaded_job_output_dir
        from app.services.video import get_video_meta as loaded_get_video_meta
        from app.services.video import video_id as loaded_video_id

        main_module = loaded_main
        build_job_report = loaded_build_job_report
        job_output_dir = loaded_job_output_dir
        get_video_meta = loaded_get_video_meta
        video_id = loaded_video_id

    def _install_fake_analyzer_if_needed(self) -> None:
        if self.real_ml:
            return
        assert main_module is not None
        self._original_analyze = main_module.analyze_videos
        main_module.analyze_videos = self._fake_analyze_videos

    def _restore_analyzer(self) -> None:
        if self._original_analyze is not None:
            assert main_module is not None
            main_module.analyze_videos = self._original_analyze

    def _find_first_file(self, directory: Path, suffixes: set[str]) -> Path | None:
        if not directory.exists():
            return None
        files = sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in suffixes)
        return files[0] if files else None

    def _run_positive_case(self, title: str, source: Path) -> None:
        job_id = ""
        videos: list[dict[str, Any]] = []
        try:
            payload = self._upload(source)
            job_id = payload["job_id"]
            videos = payload["videos"]
            self._require(len(videos) > 0, f"{title}: список видео", "сервер вернул пустой список видео")
            self._record(f"{title}: загрузка", True, f"job_id={job_id}, videos={len(videos)}")
            self._run_uploaded_job_api_checks(title, job_id, videos)
        except Exception as exc:
            self._record(f"{title}: загрузка", False, str(exc))
            return

        for item in videos:
            vid = item["video_id"]
            try:
                frame_response = main_module.get_frame(job_id, vid, frame=0)
                frame_path = Path(frame_response.path)
                self._require(frame_path.exists(), f"{title}: первый кадр {vid}", str(frame_path))
                self._require(frame_response.media_type == "image/jpeg", f"{title}: тип первого кадра {vid}", frame_response.media_type or "")
                self._record(f"{title}: первый кадр {vid}", True, f"получен JPEG: {frame_path.name}")
            except Exception as exc:
                self._record(f"{title}: первый кадр {vid}", False, str(exc))

        process_response: dict[str, Any] | None = None
        try:
            process_response, status = self._process_and_wait(job_id, videos)
            self._require(process_response["status"] == "processing", f"{title}: запуск обработки", str(process_response))
            self._record(f"{title}: запуск обработки", True, "статус processing")
            self._require(status["status"] == "done", f"{title}: завершение обработки", str(status))
            self._record(f"{title}: завершение обработки", True, "статус done")
        except Exception as exc:
            failed_step = "запуск обработки" if process_response is None else "завершение обработки"
            self._record(f"{title}: {failed_step}", False, self._error_with_processing_log(exc))
            return

        try:
            report = main_module.get_report(job_id)
            overview = {row["metric"]: row["value"] for row in report.get("overview", [])}
            self._require(overview.get("videos_total") == len(videos), f"{title}: метрики отчета", str(overview))
            self._require("html_report_url" in report, f"{title}: ссылки отчета", str(report.keys()))
            self._record(f"{title}: отчет", True, f"videos_total={overview.get('videos_total')}")
        except Exception as exc:
            self._record(f"{title}: отчет", False, str(exc))

        try:
            download_response = main_module.download_report(job_id)
            zip_path = Path(download_response.path)
            self._require(zip_path.exists(), f"{title}: скачивание ZIP", str(zip_path))
            self._require(zip_path.stat().st_size > 0, f"{title}: ZIP не пустой", str(zip_path))
            self._record(f"{title}: скачивание ZIP", True, f"{zip_path.stat().st_size} bytes")
        except Exception as exc:
            self._record(f"{title}: скачивание ZIP", False, str(exc))

    def _run_uploaded_job_api_checks(self, title: str, job_id: str, videos: list[dict[str, Any]]) -> None:
        assert main_module is not None
        try:
            job = main_module.get_job(job_id)
            self._require(job["job_id"] == job_id, f"{title}: GET /api/jobs", str(job))
            self._require(job["status"] == "uploaded", f"{title}: статус после загрузки", str(job))
            self._require(len(job["items"]) == len(videos), f"{title}: items после загрузки", str(job.get("items")))
            self._record(f"{title}: API GET /api/jobs", True, "возвращает uploaded job с items")
        except Exception as exc:
            self._record(f"{title}: API GET /api/jobs", False, str(exc))

        try:
            status = main_module.job_status(job_id)
            self._require(status["status"] == "uploaded", f"{title}: GET /status до обработки", str(status))
            self._require(status["progress"] == 0, f"{title}: progress до обработки", str(status))
            self._record(f"{title}: API GET /status до обработки", True, "status=uploaded, progress=0")
        except Exception as exc:
            self._record(f"{title}: API GET /status до обработки", False, str(exc))

        try:
            main_module.get_report(job_id)
            self._record(f"{title}: API /report до обработки", False, "отчет не должен быть доступен до обработки")
        except HTTPException as exc:
            ok = exc.status_code == 409 and "not ready" in str(exc.detail).lower()
            self._record(f"{title}: API /report до обработки", ok, f"HTTP {exc.status_code}: {exc.detail}")
        except Exception as exc:
            self._record(f"{title}: API /report до обработки", False, str(exc))

    def _process_and_wait(self, job_id: str, videos: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
        assert main_module is not None
        payload = {"rois": self._build_rois(videos), "params": {"render_video": False}}
        self._last_processing_log = ""

        if self.verbose_ml:
            process_response = main_module.process_job(job_id, payload)
            status = self._wait_done(job_id)
            return process_response, status

        captured = StringIO()
        with redirect_stdout(captured), redirect_stderr(captured):
            process_response = main_module.process_job(job_id, payload)
            status = self._wait_done(job_id)
        self._last_processing_log = captured.getvalue()
        return process_response, status

    def _error_with_processing_log(self, exc: Exception) -> str:
        detail = str(exc)
        tail = self._processing_log_tail()
        if tail:
            detail = f"{detail}\nПоследние строки ML-лога:\n{tail}"
        return detail

    def _processing_log_tail(self, max_lines: int = 80) -> str:
        lines = [line for line in self._last_processing_log.splitlines() if line.strip()]
        return "\n".join(lines[-max_lines:])

    def _upload(self, source: Path):
        assert main_module is not None
        with source.open("rb") as file_obj:
            upload = UploadFile(filename=source.name, file=file_obj)
            return asyncio.run(main_module.create_job([upload]))

    def _build_rois(self, videos: list[dict[str, Any]]) -> dict[str, list[list[float]]]:
        rois: dict[str, list[list[float]]] = {}
        for item in videos:
            meta = item.get("meta") or {}
            width = float(meta.get("width") or 640)
            height = float(meta.get("height") or 480)
            x1, y1 = width * 0.2, height * 0.2
            x2, y2 = width * 0.8, height * 0.8
            rois[item["video_id"]] = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        return rois

    def _wait_done(self, job_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout_seconds
        last_status: dict[str, Any] = {}
        while time.monotonic() < deadline:
            assert main_module is not None
            last_status = main_module.job_status(job_id)
            if last_status.get("status") in {"done", "error"}:
                return last_status
            time.sleep(POLL_INTERVAL_SECONDS)
        return last_status

    def _run_invalid_file_case(self) -> None:
        upload = UploadFile(filename="not_video.txt", file=BytesIO(b"not a video"))
        response = self._call_create_job_expect_error(upload)
        ok = response.status_code == 400 and "Не найдено видео" in str(response.detail)
        self._record("Негативный тест: неподдерживаемый файл", ok, f"HTTP {response.status_code}: {response.detail}")

    def _run_zip_without_video_case(self) -> None:
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as archive:
            archive.writestr("readme.txt", "no video here")
        zip_buffer.seek(0)

        upload = UploadFile(filename="empty_dataset.zip", file=zip_buffer)
        response = self._call_create_job_expect_error(upload)
        ok = response.status_code == 400 and "Не найдено видео" in str(response.detail)
        self._record("Негативный тест: ZIP без видео", ok, f"HTTP {response.status_code}: {response.detail}")

    def _run_missing_job_api_cases(self) -> None:
        assert main_module is not None
        cases: list[tuple[str, Callable[[], Any], int]] = [
            ("API: GET неизвестного job", lambda: main_module.get_job("missing-job"), 404),
            ("API: status неизвестного job", lambda: main_module.job_status("missing-job"), 404),
            ("API: process неизвестного job", lambda: main_module.process_job("missing-job", {"rois": {}}), 404),
            ("API: download неизвестного job", lambda: main_module.download_report("missing-job"), 404),
        ]
        for name, call, expected_status in cases:
            try:
                call()
                self._record(name, False, f"ожидался HTTP {expected_status}")
            except HTTPException as exc:
                self._record(name, exc.status_code == expected_status, f"HTTP {exc.status_code}: {exc.detail}")
            except Exception as exc:
                self._record(name, False, str(exc))

    def _call_create_job_expect_error(self, upload: UploadFile) -> HTTPException:
        try:
            assert main_module is not None
            asyncio.run(main_module.create_job([upload]))
        except HTTPException as exc:
            return exc
        raise AssertionError("Сервер принял файл, который должен быть отклонен")

    def _fake_analyze_videos(
        self,
        job_id: str,
        videos: list[Path],
        rois: dict[str, list[list[float]]],
        params: dict[str, Any] | None = None,
        progress_callback=None,
    ) -> dict[str, Any]:
        assert build_job_report is not None
        assert get_video_meta is not None
        assert job_output_dir is not None
        assert video_id is not None
        output_dir = job_output_dir(job_id)
        results: list[dict[str, Any]] = []
        if progress_callback:
            progress_callback(0.1, "Тестовый анализ", "Подготовка тестовых результатов")
        for video_path in videos:
            vid = video_id(video_path)
            video_dir = output_dir / "pipeline" / vid
            video_dir.mkdir(parents=True, exist_ok=True)
            summary_image = video_dir / f"{vid}_summary.jpg"
            summary_image.write_bytes(ONE_PIXEL_JPEG)
            zones_df = pd.DataFrame(
                [
                    {
                        "zone_id": 1,
                        "rank": 1,
                        "top_action": "reach_to_shelf",
                        "interest_0_100": 91.0,
                        "raw_interest": 0.91,
                        "take_score": 1.0,
                        "return_score": 0.0,
                        "inspect_score": 0.0,
                        "business_comment": "Тестовая зона интереса сформирована автоматически.",
                    }
                ]
            )
            events_df = pd.DataFrame(
                [
                    {
                        "zone_id": 1,
                        "zone_top_action": "reach_to_shelf",
                        "interest_score": 0.91,
                        "extension_norm": 0.55,
                        "roi_points": len(rois.get(vid) or rois.get("__global__") or []),
                    }
                ]
            )
            results.append(
                {
                    "video_id": vid,
                    "status": "ok",
                    "video_path": str(video_path),
                    "meta": get_video_meta(video_path),
                    "paths": {"summary_image": summary_image},
                    "zones_df": zones_df,
                    "events_df": events_df,
                }
            )
        report = build_job_report(job_id, output_dir, results)
        return {"results": results, "report": report}

    def _require(self, condition: bool, name: str, detail: str) -> None:
        if not condition:
            raise AssertionError(f"{name}: {detail}")

    def _record(self, name: str, ok: bool, detail: str) -> None:
        self.results.append(CheckResult(name=name, ok=ok, detail=detail))
        icon = "✅" if ok else "❌"
        print(f"{icon} {name} — {detail}")

    def _print_summary(self) -> None:
        passed = sum(1 for item in self.results if item.ok)
        failed = len(self.results) - passed
        print("=== Сводка автоматического тестирования сервиса ===")
        print(f"Всего проверок: {len(self.results)}")
        print(f"Успешно: {passed}")
        print(f"Ошибок: {failed}")
        for item in self.results:
            icon = "✅" if item.ok else "❌"
            print(f"{icon} {item.name}: {item.detail}")

    def _write_test_report(self) -> None:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = self.report_dir / f"service_test_report_{stamp}"
        passed = sum(1 for item in self.results if item.ok)
        failed = len(self.results) - passed
        payload = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "assets_dir": str(self.assets_dir),
            "real_ml": self.real_ml,
            "verbose_ml": self.verbose_ml,
            "total": len(self.results),
            "passed": passed,
            "failed": failed,
            "results": [asdict(item) for item in self.results],
        }
        json_path = base.with_suffix(".json")
        html_path = base.with_suffix(".html")
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        html_path.write_text(self._render_html_report(payload), encoding="utf-8")
        print(f"Отчет о тестировании: {html_path}")
        print(f"JSON-отчет: {json_path}")

    def _render_html_report(self, payload: dict[str, Any]) -> str:
        from html import escape

        result_rows = []
        for idx, item in enumerate(self.results, start=1):
            status = "✅" if item.ok else "❌"
            result_rows.append(
                f"<tr><td>{idx}</td><td>{status}</td><td>{escape(item.name)}</td>"
                f"<td>{escape(item.detail).replace(chr(10), '<br>')}</td></tr>"
            )
        return f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Отчет автоматического тестирования</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 32px; color: #172033; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #d8deea; padding: 8px; text-align: left; }}
th {{ background: #edf3fb; }}
</style>
</head>
<body>
<h1>Отчет автоматического тестирования сервиса</h1>
<ul>
<li>Дата запуска: {escape(payload['created_at'])}</li>
<li>Всего проверок: {payload['total']}</li>
<li>Успешно: {payload['passed']}</li>
<li>Ошибок: {payload['failed']}</li>
<li>Реальный ML-пайплайн: {payload['real_ml']}</li>
</ul>
<table>
<thead><tr><th>№</th><th>Статус</th><th>Проверка</th><th>Детали</th></tr></thead>
<tbody>{''.join(result_rows)}</tbody>
</table>
</body>
</html>
"""

    def _print_missing_assets(self, video: Path | None, archive: Path | None) -> None:
        print("❌ Не хватает тестовых данных.")
        if video is None:
            print(f"- Добавьте хотя бы одно видео в {self.assets_dir / 'videos'}")
        if archive is None:
            print(f"- Добавьте хотя бы один ZIP-архив с видео в {self.assets_dir / 'archives'}")
        print("\nПосле добавления файлов запустите команду:")
        print("python scripts/run_service_tests.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Автоматическая проверка сервиса Retail Interest Zones.")
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=DEFAULT_ASSETS_DIR,
        help="Папка с подпапками videos/ и archives/.",
    )
    parser.add_argument(
        "--real-ml",
        action="store_true",
        help="Использовать настоящий ML-пайплайн вместо тестовой заглушки.",
    )
    parser.add_argument(
        "--verbose-ml",
        action="store_true",
        help="Показывать подробный stdout/stderr ML-пайплайна во время обработки.",
    )
    parser.add_argument(
        "--clean-data",
        action="store_true",
        help="Перед запуском удалить data/uploads и data/outputs, созданные предыдущими проверками.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Папка для HTML/JSON-отчетов о тестовом прогоне.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=None,
        help="Максимальное время ожидания обработки; для --real-ml по умолчанию 21600 секунд.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.clean_data:
        shutil.rmtree(ROOT_DIR / "data" / "uploads", ignore_errors=True)
        shutil.rmtree(ROOT_DIR / "data" / "outputs", ignore_errors=True)
    runner = ServiceTestRunner(
        assets_dir=args.assets_dir,
        real_ml=args.real_ml,
        verbose_ml=args.verbose_ml,
        report_dir=args.report_dir,
        timeout_seconds=args.timeout_seconds,
    )
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
