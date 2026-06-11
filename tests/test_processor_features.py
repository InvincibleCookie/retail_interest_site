from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd

sys.modules.setdefault("cv2", types.SimpleNamespace())

from app.services import processor


def test_processor_extracts_features_before_running_core(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"video")
    feature_paths = {
        "pose_chunk": tmp_path / "pose.npz",
        "rgb_chunk": tmp_path / "rgb.npz",
        "rtmw_frame": tmp_path / "rtmw.npz",
        "normalizer": tmp_path / "normalizer.npz",
    }
    calls = {}

    class FakeCore:
        OUT_DIR = tmp_path

        @staticmethod
        def predict_actions_c6(*args, **kwargs):
            calls["predict"] = kwargs
            return pd.DataFrame([
                {"frame_start": 0, "frame_end": 5, "zone_action_signal": 0.30},
                {"frame_start": 6, "frame_end": 11, "zone_action_signal": 0.01},
            ])

        @staticmethod
        def analyze_interest_zones_v5(**kwargs):
            calls["core"] = kwargs
            return {"zones_df": pd.DataFrame(), "event_points_df": pd.DataFrame(), "paths": {}}

    def fake_extract(video_path, output_dir, params=None, progress_callback=None):
        calls["extract"] = (video_path, output_dir, params)
        if progress_callback:
            progress_callback(0.5, "YOLOv8m-pose", "Кадр 5 из 10")
        return feature_paths

    monkeypatch.setattr(processor, "_load_core", lambda: FakeCore)

    def fake_extract_rtmw(video_path, frame_path, frame_indices, **kwargs):
        calls["rtmw"] = (video_path, frame_path, list(frame_indices), kwargs)
        callback = kwargs.get("progress_callback")
        if callback:
            callback(1.0, "RTMW WholeBody", "Кадры обработаны")

    monkeypatch.setattr(processor, "extract_classification_features", fake_extract)
    monkeypatch.setattr(processor, "extract_rtmw", fake_extract_rtmw)
    monkeypatch.setattr(processor, "job_output_dir", lambda job_id: tmp_path / job_id)
    monkeypatch.setattr(processor, "get_video_meta", lambda path: {"width": 640, "height": 480, "n_frames": 12})
    monkeypatch.setattr(processor, "build_job_report", lambda job_id, output_dir, results: {"overview": []})

    progress = []
    result = processor.analyze_videos(
        "job",
        [video],
        {"sample": [[0, 0], [1, 0], [1, 1]]},
        progress_callback=lambda fraction, stage, message: progress.append((fraction, stage, message)),
    )

    assert calls["extract"][0] == video
    assert calls["predict"]["pose_chunk_path"] == feature_paths["pose_chunk"]
    assert calls["rtmw"][2] == list(range(6))
    assert calls["core"]["pose_chunk_path"] == feature_paths["pose_chunk"]
    assert calls["core"]["rgb_chunk_path"] == feature_paths["rgb_chunk"]
    assert calls["core"]["rtmw_frame_path"] == feature_paths["rtmw_frame"]
    assert calls["core"]["normalizer_path"] == feature_paths["normalizer"]
    assert calls["core"]["pred_df"] is not None
    assert any(stage == "YOLOv8m-pose" for _, stage, _ in progress)
    assert progress[-1][0] == 1.0
    assert result["report"] == {"overview": []}


def test_select_rtmw_frame_indices_uses_only_active_chunks() -> None:
    predictions = pd.DataFrame([
        {"frame_start": -2, "frame_end": 3, "zone_action_signal": 0.20},
        {"frame_start": 4, "frame_end": 9, "zone_action_signal": 0.01},
        {"frame_start": 10, "frame_end": 20, "zone_action_signal": 0.30},
    ])

    selected = processor.select_rtmw_frame_indices(predictions, n_frames=14, min_action_signal=0.05)

    assert selected == [0, 1, 2, 3, 10, 11, 12, 13]
