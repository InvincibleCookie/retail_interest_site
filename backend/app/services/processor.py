from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from app.services.video import video_id, get_video_meta
from app.services.reports import enrich_zone_meaning, build_job_report
from app.services.storage import job_output_dir
from app.services.feature_extraction import extract_classification_features, extract_rtmw


def _load_core():
    from app.pipeline import interest_zones_core as core
    return core


DEFAULT_PARAMS = {
    "render_video": False,
    "max_render_frames": None,
    "raw_score_thr": 0.05,
    "raw_min_action_signal": 0.05,
    "raw_min_extension": 0.18,
    "event_min_extension": 0.40,
    "event_min_action_signal": 0.18,
    "min_event_score": 0.030,
    "max_gap_frames": 16,
    "spatial_jump_px": 95,
    "peak_top_k": 3,
    "projection_alpha": 0.16,
    "heatmap_radius_px": 30,
    "heatmap_blur_px": 0,
    "zone_relative_thr": 0.36,
    "zone_percentile_thr": 86,
    "zone_merge_dilate_px": 18,
    "min_component_area": 160,
}


def select_rtmw_frame_indices(
    predictions: pd.DataFrame,
    n_frames: int,
    min_action_signal: float,
) -> list[int]:
    selected: set[int] = set()
    active = predictions[predictions["zone_action_signal"] >= float(min_action_signal)]
    for row in active.itertuples(index=False):
        start = max(0, int(row.frame_start))
        end = min(n_frames - 1, int(row.frame_end))
        if end >= start:
            selected.update(range(start, end + 1))
    return sorted(selected)


def analyze_videos(
    job_id: str,
    videos: list[Path],
    rois: dict[str, list[list[float]]],
    params: dict[str, Any] | None = None,
    progress_callback: Callable[[float, str, str], None] | None = None,
) -> dict[str, Any]:
    core = _load_core()
    output_dir = job_output_dir(job_id)
    core.OUT_DIR = output_dir / "pipeline"
    core.OUT_DIR.mkdir(parents=True, exist_ok=True)

    final_params = DEFAULT_PARAMS.copy()
    if params:
        for k, v in params.items():
            if k in final_params:
                final_params[k] = v

    results: list[dict[str, Any]] = []
    total_videos = max(1, len(videos))

    def report(video_index: int, local_fraction: float, stage: str, message: str) -> None:
        if progress_callback is None:
            return
        video_fraction = (video_index + local_fraction) / total_videos
        progress_callback(video_fraction * 0.95, stage, message)

    for video_index, video_path in enumerate(videos):
        vid = video_id(video_path)
        roi = rois.get(vid) or rois.get("__global__")
        try:
            report(video_index, 0.0, "Подготовка видео", f"{vid}: запуск извлечения признаков")
            feature_paths = extract_classification_features(
                video_path,
                output_dir,
                params=params,
                progress_callback=lambda fraction, stage, message, i=video_index: report(
                    i,
                    fraction * 0.55,
                    stage,
                    f"{vid}: {message}",
                ),
            )
            report(video_index, 0.56, "Классификация действий", f"{vid}: запуск C6 LateFusion Gated BiGRU")
            pred_df = core.predict_actions_c6(
                video_path,
                pose_chunk_path=feature_paths["pose_chunk"],
                rgb_chunk_path=feature_paths["rgb_chunk"],
                normalizer_path=feature_paths["normalizer"],
            )
            meta = get_video_meta(video_path)
            selected_frames = select_rtmw_frame_indices(
                pred_df,
                meta["n_frames"],
                final_params["raw_min_action_signal"],
            )
            report(
                video_index,
                0.63,
                "Выбор кадров RTMW",
                f"{vid}: для локализации кистей выбрано {len(selected_frames)} из {meta['n_frames']} кадров",
            )
            extract_rtmw(
                video_path,
                feature_paths["rtmw_frame"],
                selected_frames,
                person_model_name=str((params or {}).get("yolo_person_model", "yolo11n.pt")),
                conf=float((params or {}).get("yolo_person_conf", 0.30)),
                imgsz=int((params or {}).get("yolo_imgsz", 640)),
                progress_callback=lambda fraction, stage, message, i=video_index: report(
                    i,
                    0.64 + fraction * 0.20,
                    stage,
                    f"{vid}: {message}",
                ),
            )
            res = core.analyze_interest_zones_v5(
                video_path=video_path,
                manual_roi_polygon=roi,
                save_debug=True,
                pose_chunk_path=feature_paths["pose_chunk"],
                rgb_chunk_path=feature_paths["rgb_chunk"],
                rtmw_frame_path=feature_paths["rtmw_frame"],
                normalizer_path=feature_paths["normalizer"],
                pred_df=pred_df,
                progress_callback=lambda fraction, stage, message, i=video_index: report(
                    i,
                    0.85 + fraction * 0.09,
                    stage,
                    f"{vid}: {message}",
                ),
                **final_params,
            )
            report(video_index, 0.94, "Построение зон", f"{vid}: формирование heatmap и зон интереса")
            zones_df = res.get("zones_df", pd.DataFrame())
            events_df = res.get("event_points_df", pd.DataFrame())
            zones_df = enrich_zone_meaning(zones_df, events_df)

            # Save enriched zones over the raw file too.
            if len(zones_df) > 0:
                zones_path = Path(res["paths"].get("zones", core.OUT_DIR / vid / f"{vid}_heatmap_zones.csv"))
                zones_df.to_csv(zones_path, index=False, encoding="utf-8-sig")

            results.append({
                "video_id": vid,
                "status": "ok",
                "video_path": str(video_path),
                "meta": get_video_meta(video_path),
                "paths": res.get("paths", {}),
                "zones_df": zones_df,
                "events_df": events_df,
            })
            report(video_index, 1.0, "Видео обработано", f"{vid}: обработка завершена")
        except Exception as e:
            results.append({
                "video_id": vid,
                "status": "error",
                "error": str(e),
                "video_path": str(video_path),
                "meta": {},
                "paths": {},
                "zones_df": pd.DataFrame(),
                "events_df": pd.DataFrame(),
            })
        finally:
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    if progress_callback is not None:
        progress_callback(0.97, "Формирование отчета", "Создание HTML, Excel, CSV и ZIP")
    report = build_job_report(job_id, output_dir, results)
    if progress_callback is not None:
        progress_callback(1.0, "Готово", "Отчет сформирован")
    return {"results": results, "report": report}
