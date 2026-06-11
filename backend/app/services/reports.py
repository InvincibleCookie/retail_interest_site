from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import math
import shutil

import numpy as np
import pandas as pd


def _safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _relative(path: str | Path, root: Path) -> str:
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return str(path)


def enrich_zone_meaning(zones_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    if zones_df is None or len(zones_df) == 0:
        return pd.DataFrame()
    zones = zones_df.copy()
    events = events_df.copy() if events_df is not None else pd.DataFrame()

    if len(events) == 0:
        zones["take_score"] = 0.0
        zones["return_score"] = 0.0
        zones["inspect_score"] = 0.0
        zones["business_comment"] = "Нет событий для интерпретации."
        return zones

    if "zone_id" not in events.columns:
        # fallback: use zone_id already computed in zones only
        events["zone_id"] = None

    action_col = "zone_top_action" if "zone_top_action" in events.columns else "top_action"
    score_col = "interest_score" if "interest_score" in events.columns else None

    extra_rows = []
    for _, z in zones.iterrows():
        zid = z.get("zone_id")
        ev = events[events["zone_id"] == zid] if zid in set(events.get("zone_id", [])) else pd.DataFrame()

        if len(ev) == 0:
            take_score = return_score = inspect_score = 0.0
            comment = "Зона найдена по тепловой карте, но событий для детализации мало."
        else:
            if score_col:
                scores = ev[score_col].astype(float)
            else:
                scores = pd.Series(np.ones(len(ev)), index=ev.index)
            actions = ev[action_col].astype(str)
            take_score = float(scores[actions.isin(["reach_to_shelf", "hand_in_shelf"])].sum())
            return_score = float(scores[actions.isin(["retract_from_shelf"])].sum())
            inspect_score = float(scores[actions.isin(["inspect_shelf", "inspect_product"])].sum())

            parts = []
            if take_score >= max(return_score, inspect_score):
                parts.append("много дотягиваний/контактов с полкой")
            if return_score > 0.35 * max(take_score, 1e-6):
                parts.append("часто фиксируется отведение руки от полки")
            if inspect_score > 0.35 * max(take_score + return_score, 1e-6):
                parts.append("заметная доля осмотра полки")
            comment = "; ".join(parts) if parts else "слабая, но стабильная активность"

        extra_rows.append({
            "zone_id": zid,
            "take_score": take_score,
            "return_score": return_score,
            "inspect_score": inspect_score,
            "business_comment": comment,
        })

    extra = pd.DataFrame(extra_rows)
    zones = zones.merge(extra, on="zone_id", how="left")
    return zones


def build_job_report(job_id: str, output_dir: Path, results: list[dict[str, Any]]) -> dict[str, Any]:
    report_dir = output_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    video_rows = []
    all_zones = []
    all_events = []

    for item in results:
        video_id = item["video_id"]
        status = item.get("status", "ok")
        error = item.get("error", "")
        paths = item.get("paths", {}) or {}
        zones_df = item.get("zones_df", pd.DataFrame())
        events_df = item.get("events_df", pd.DataFrame())
        meta = item.get("meta", {}) or {}

        if zones_df is not None and len(zones_df) > 0:
            z = zones_df.copy()
            z.insert(0, "video_id", video_id)
            all_zones.append(z)
        if events_df is not None and len(events_df) > 0:
            e = events_df.copy()
            e.insert(0, "video_id", video_id)
            all_events.append(e)

        top = zones_df.sort_values("rank").iloc[0] if zones_df is not None and len(zones_df) > 0 and "rank" in zones_df.columns else None
        image_rel = ""
        if paths.get("summary_image"):
            p = Path(paths["summary_image"])
            if p.exists():
                image_rel = _relative(p, output_dir)

        video_rows.append({
            "video_id": video_id,
            "status": status,
            "error": error,
            "width": meta.get("width"),
            "height": meta.get("height"),
            "fps": meta.get("fps"),
            "n_frames": meta.get("n_frames"),
            "n_zones": 0 if zones_df is None else len(zones_df),
            "n_events": 0 if events_df is None else len(events_df),
            "top_zone": None if top is None else top.get("zone_id"),
            "top_action": None if top is None else top.get("top_action"),
            "top_interest_0_100": 0.0 if top is None else _safe_float(top.get("interest_0_100")),
            "summary_image": image_rel,
        })

    video_summary = pd.DataFrame(video_rows)
    all_zones_df = pd.concat(all_zones, ignore_index=True) if all_zones else pd.DataFrame()
    all_events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()

    action_summary = pd.DataFrame()
    if len(all_events_df) > 0:
        action_col = "zone_top_action" if "zone_top_action" in all_events_df.columns else None
        if action_col:
            action_summary = (
                all_events_df.groupby(action_col)
                .agg(n_events=(action_col, "size"), mean_score=("interest_score", "mean"), mean_extension=("extension_norm", "mean"))
                .reset_index()
                .sort_values("n_events", ascending=False)
            )
            action_summary["event_share"] = action_summary["n_events"] / max(1, action_summary["n_events"].sum())

    top_zones = pd.DataFrame()
    if len(all_zones_df) > 0:
        top_zones = all_zones_df.sort_values("raw_interest", ascending=False).head(100).reset_index(drop=True)

    overview = pd.DataFrame([
        {"metric": "videos_total", "value": len(video_summary)},
        {"metric": "processed_ok", "value": int((video_summary["status"] == "ok").sum()) if len(video_summary) else 0},
        {"metric": "failed", "value": int((video_summary["status"] != "ok").sum()) if len(video_summary) else 0},
        {"metric": "total_event_points", "value": int(video_summary["n_events"].fillna(0).sum()) if len(video_summary) else 0},
        {"metric": "total_zones", "value": int(video_summary["n_zones"].fillna(0).sum()) if len(video_summary) else 0},
    ])

    # Save data
    overview.to_csv(report_dir / "overview.csv", index=False, encoding="utf-8-sig")
    video_summary.to_csv(report_dir / "video_summary.csv", index=False, encoding="utf-8-sig")
    all_zones_df.to_csv(report_dir / "all_zones.csv", index=False, encoding="utf-8-sig")
    all_events_df.to_csv(report_dir / "all_event_points.csv", index=False, encoding="utf-8-sig")
    action_summary.to_csv(report_dir / "action_summary.csv", index=False, encoding="utf-8-sig")
    top_zones.to_csv(report_dir / "top_zones.csv", index=False, encoding="utf-8-sig")

    excel_path = report_dir / "retail_interest_report.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        overview.to_excel(writer, sheet_name="overview", index=False)
        video_summary.to_excel(writer, sheet_name="videos", index=False)
        action_summary.to_excel(writer, sheet_name="actions", index=False)
        top_zones.to_excel(writer, sheet_name="top_zones", index=False)
        all_zones_df.to_excel(writer, sheet_name="all_zones", index=False)
        # Excel has row limits; save only first 100k events.
        all_events_df.head(100000).to_excel(writer, sheet_name="events_sample", index=False)

    html_path = report_dir / "retail_interest_report.html"
    html = render_html_report(job_id, overview, video_summary, action_summary, top_zones, output_dir)
    html_path.write_text(html, encoding="utf-8")

    zip_path = output_dir / f"retail_interest_report_{job_id}.zip"
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", output_dir)

    return {
        "overview": overview.to_dict(orient="records"),
        "video_summary": video_summary.to_dict(orient="records"),
        "action_summary": action_summary.to_dict(orient="records"),
        "top_zones": top_zones.head(50).to_dict(orient="records"),
        "html_report": str(html_path),
        "excel_report": str(excel_path),
        "zip_report": str(zip_path),
        "html_report_url": f"/outputs/{job_id}/report/retail_interest_report.html",
        "excel_report_url": f"/outputs/{job_id}/report/retail_interest_report.xlsx",
        "overview_csv_url": f"/outputs/{job_id}/report/overview.csv",
        "video_summary_csv_url": f"/outputs/{job_id}/report/video_summary.csv",
        "all_zones_csv_url": f"/outputs/{job_id}/report/all_zones.csv",
        "all_events_csv_url": f"/outputs/{job_id}/report/all_event_points.csv",
    }


def render_html_report(job_id: str, overview: pd.DataFrame, videos: pd.DataFrame, actions: pd.DataFrame, top_zones: pd.DataFrame, output_dir: Path) -> str:
    def table(df: pd.DataFrame, rows=40) -> str:
        if df is None or len(df) == 0:
            return "<p class='muted'>Нет данных.</p>"
        return df.head(rows).to_html(index=False, classes="data", escape=False)

    cards = []
    for _, r in videos.iterrows():
        img = r.get("summary_image", "")
        img_html = f"<img src='../{img}' alt='summary'>" if isinstance(img, str) and img else "<div class='no-img'>no image</div>"
        cards.append(f"""
        <article class='card'>
          <div class='thumb'>{img_html}</div>
          <div class='card-body'>
            <h3>{r.get('video_id')}</h3>
            <p><b>Статус:</b> {r.get('status')}</p>
            <p><b>События:</b> {r.get('n_events')} · <b>Зоны:</b> {r.get('n_zones')}</p>
            <p><b>Главная зона:</b> {r.get('top_zone')} · <b>действие:</b> {r.get('top_action')}</p>
            <p><b>Интерес:</b> {r.get('top_interest_0_100')}</p>
            <p class='error'>{r.get('error') or ''}</p>
          </div>
        </article>
        """)

    return f"""
<!doctype html>
<html lang='ru'>
<head>
  <meta charset='utf-8'>
  <title>Retail Interest Report {job_id}</title>
  <style>
    body {{ margin:0; font-family: Inter, Segoe UI, Arial, sans-serif; background:#f6f7fb; color:#152238; }}
    header {{ padding:36px 44px; background:linear-gradient(135deg,#10233f,#285c8f); color:white; }}
    h1 {{ margin:0 0 8px; font-size:34px; }}
    h2 {{ margin-top:34px; }}
    main {{ padding:28px 44px; }}
    .muted {{ color:#65748b; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); gap:18px; }}
    .card {{ background:white; border-radius:18px; overflow:hidden; box-shadow:0 10px 30px rgba(21,34,56,.08); border:1px solid #e5e9f2; }}
    .thumb img {{ width:100%; display:block; }}
    .no-img {{ min-height:220px; display:flex; align-items:center; justify-content:center; color:#8b98aa; background:#eef1f6; }}
    .card-body {{ padding:18px; }}
    .card h3 {{ margin:0 0 10px; }}
    table.data {{ border-collapse:collapse; width:100%; background:white; border-radius:14px; overflow:hidden; box-shadow:0 8px 24px rgba(21,34,56,.06); }}
    table.data th, table.data td {{ border-bottom:1px solid #edf0f5; padding:10px 12px; text-align:left; font-size:14px; }}
    table.data th {{ background:#eef4fb; }}
    .error {{ color:#bd2f2f; }}
  </style>
</head>
<body>
<header>
  <h1>Сводный отчет по зонам интереса</h1>
  <p>Job: {job_id}. Зоны построены по heatmap контактных событий руки у полки.</p>
</header>
<main>
  <h2>Общие показатели</h2>
  {table(overview, 20)}

  <h2>Действия, формирующие интерес</h2>
  <p class='muted'>reach_to_shelf и hand_in_shelf интерпретируются как сильный интерес/контакт с товарной зоной. retract_from_shelf — как отведение руки от полки; это может быть взятие товара или возврат, поэтому в отчете используется осторожная формулировка.</p>
  {table(actions, 20)}

  <h2>Самые активные зоны по всем видео</h2>
  {table(top_zones, 50)}

  <h2>Видео</h2>
  <section class='grid'>{''.join(cards)}</section>
</main>
</body>
</html>
"""
