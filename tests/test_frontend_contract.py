from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = (ROOT / "backend" / "app" / "static" / "index.html").read_text(encoding="utf-8")
APP_JS = (ROOT / "backend" / "app" / "static" / "app.js").read_text(encoding="utf-8")


def test_upload_button_calls_create_job_api_and_renders_video_select() -> None:
    assert 'id="uploadBtn"' in INDEX_HTML
    assert "$('uploadBtn').onclick" in APP_JS
    assert "fetch('/api/jobs'" in APP_JS
    assert "method:'POST'" in APP_JS
    assert "jobId = data.job_id" in APP_JS
    assert "videos = data.videos" in APP_JS
    assert "videoSelect" in APP_JS


def test_process_button_sends_roi_params_and_starts_status_polling() -> None:
    assert 'id="processBtn"' in INDEX_HTML
    assert "$('processBtn').onclick" in APP_JS
    assert "Нарисуйте ROI полки" in APP_JS
    assert "payloadRois['__global__']" in APP_JS
    assert "body: JSON.stringify({rois: payloadRois, params: processParams()})" in APP_JS
    assert "pollStatus();" in APP_JS


def test_report_buttons_are_bound_to_report_api_results() -> None:
    assert 'id="downloadLink"' in INDEX_HTML
    assert "downloadLink" in APP_JS
    assert "/download" in APP_JS
    assert "html_report_url" in APP_JS
    assert "excel_report_url" in APP_JS
    assert "all_zones_csv_url" in APP_JS
    assert "all_events_csv_url" in APP_JS


def test_apply_all_button_copies_current_roi_to_all_videos() -> None:
    assert 'id="applyAllBtn"' in INDEX_HTML
    assert "$('applyAllBtn').onclick" in APP_JS
    assert "videos.forEach(v => rois[v.video_id]" in APP_JS
    assert "ROI применена ко всем видео" in APP_JS


def test_preview_frame_is_fetched_and_errors_are_displayed() -> None:
    assert "async function loadFrame(vid)" in APP_JS
    assert "await fetch(url, {cache: 'no-store'})" in APP_JS
    assert "await response.blob()" in APP_JS
    assert "URL.createObjectURL(blob)" in APP_JS
    assert "Не удалось загрузить первый кадр" in APP_JS
    assert "await loadFrame(videos[0].video_id)" in APP_JS


def test_static_assets_are_versioned_to_avoid_stale_preview_code() -> None:
    assert '/static/app.js?v=' in INDEX_HTML
    assert '/static/styles.css?v=' in INDEX_HTML


def test_processing_progress_shows_stage_device_elapsed_time_and_eta() -> None:
    for element_id in ["processStage", "progressPercent", "processDevice", "processElapsed", "processEta"]:
        assert f'id="{element_id}"' in INDEX_HTML
    assert "function formatDuration(seconds)" in APP_JS
    assert "data.eta_seconds" in APP_JS
    assert "data.elapsed_seconds" in APP_JS
    assert "data.device" in APP_JS
    assert "data.stage" in APP_JS
