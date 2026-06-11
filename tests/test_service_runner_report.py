from __future__ import annotations

import json
from pathlib import Path

from scripts.run_service_tests import CheckResult, ServiceTestRunner


def test_service_runner_writes_html_and_json_report(tmp_path: Path) -> None:
    runner = ServiceTestRunner(
        assets_dir=tmp_path / "assets",
        real_ml=False,
        verbose_ml=False,
        report_dir=tmp_path / "reports",
    )
    runner.results = [
        CheckResult("API: кнопка загрузки", True, "POST /api/jobs"),
        CheckResult("API: ошибка", False, "HTTP 500"),
    ]

    runner._write_test_report()

    html_reports = list((tmp_path / "reports").glob("service_test_report_*.html"))
    json_reports = list((tmp_path / "reports").glob("service_test_report_*.json"))
    assert len(html_reports) == 1
    assert len(json_reports) == 1

    html = html_reports[0].read_text(encoding="utf-8")
    assert "Отчет автоматического тестирования сервиса" in html
    assert "API: кнопка загрузки" in html
    assert "Ошибок: 1" in html

    payload = json.loads(json_reports[0].read_text(encoding="utf-8"))
    assert payload["total"] == 2
    assert payload["passed"] == 1
    assert payload["failed"] == 1
    assert payload["results"][0]["detail"] == "POST /api/jobs"
