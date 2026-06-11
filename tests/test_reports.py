from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from app.services.reports import build_job_report, enrich_zone_meaning


def test_enrich_zone_meaning_adds_business_scores_and_comments() -> None:
    zones = pd.DataFrame(
        [
            {"zone_id": 1, "rank": 1, "top_action": "reach_to_shelf", "interest_0_100": 90.0},
            {"zone_id": 2, "rank": 2, "top_action": "inspect_shelf", "interest_0_100": 40.0},
        ]
    )
    events = pd.DataFrame(
        [
            {"zone_id": 1, "zone_top_action": "reach_to_shelf", "interest_score": 0.7},
            {"zone_id": 1, "zone_top_action": "hand_in_shelf", "interest_score": 0.3},
            {"zone_id": 1, "zone_top_action": "retract_from_shelf", "interest_score": 0.5},
            {"zone_id": 2, "zone_top_action": "inspect_shelf", "interest_score": 0.4},
        ]
    )

    enriched = enrich_zone_meaning(zones, events)

    zone_1 = enriched[enriched["zone_id"] == 1].iloc[0]
    zone_2 = enriched[enriched["zone_id"] == 2].iloc[0]
    assert zone_1["take_score"] == 1.0
    assert zone_1["return_score"] == 0.5
    assert "дотягиваний" in zone_1["business_comment"]
    assert zone_2["inspect_score"] == 0.4
    assert "осмотра" in zone_2["business_comment"]


def test_build_job_report_creates_csv_excel_html_and_zip(tmp_path: Path) -> None:
    summary_image = tmp_path / "pipeline" / "video_1" / "summary.jpg"
    summary_image.parent.mkdir(parents=True)
    summary_image.write_bytes(b"fake-image")

    zones = pd.DataFrame(
        [
            {
                "zone_id": 1,
                "rank": 1,
                "top_action": "reach_to_shelf",
                "interest_0_100": 87.5,
                "raw_interest": 0.875,
            }
        ]
    )
    events = pd.DataFrame(
        [
            {
                "zone_id": 1,
                "zone_top_action": "reach_to_shelf",
                "interest_score": 0.75,
                "extension_norm": 0.5,
            }
        ]
    )

    report = build_job_report(
        "job-report",
        tmp_path,
        [
            {
                "video_id": "video_1",
                "status": "ok",
                "meta": {"width": 640, "height": 480, "fps": 25.0, "n_frames": 100},
                "paths": {"summary_image": summary_image},
                "zones_df": zones,
                "events_df": events,
            },
            {
                "video_id": "video_2",
                "status": "error",
                "error": "missing features",
                "meta": {},
                "paths": {},
                "zones_df": pd.DataFrame(),
                "events_df": pd.DataFrame(),
            },
        ],
    )

    assert report["overview"] == [
        {"metric": "videos_total", "value": 2},
        {"metric": "processed_ok", "value": 1},
        {"metric": "failed", "value": 1},
        {"metric": "total_event_points", "value": 1},
        {"metric": "total_zones", "value": 1},
    ]
    assert report["video_summary"][0]["summary_image"] == "pipeline/video_1/summary.jpg"
    assert Path(report["html_report"]).exists()
    assert Path(report["excel_report"]).exists()
    assert Path(report["zip_report"]).exists()
    assert (tmp_path / "report" / "overview.csv").exists()
    assert (tmp_path / "report" / "all_event_points.csv").exists()

    with ZipFile(report["zip_report"]) as archive:
        names = set(archive.namelist())
    assert "report/retail_interest_report.html" in names
    assert "report/overview.csv" in names
