from __future__ import annotations

import sys
import types

import numpy as np

sys.modules.setdefault("cv2", types.SimpleNamespace())

from app.services.feature_extraction import (
    RTMW_FRAME_DIM,
    YOLO_POSE_DIM,
    _add_pose_distances,
    _aggregate_chunks,
    _rtmw_feature,
    build_fusion_normalizer,
    extract_rtmw,
)


def test_chunk_aggregation_has_expected_pose_and_rgb_dimensions() -> None:
    pose = np.arange(12 * YOLO_POSE_DIM, dtype=np.float32).reshape(12, YOLO_POSE_DIM)
    rgb = np.arange(12 * 1280, dtype=np.float32).reshape(12, 1280)

    pose_chunks = _aggregate_chunks(pose)
    rgb_chunks = _aggregate_chunks(rgb)

    assert pose_chunks.shape == (2, 272)
    assert rgb_chunks.shape == (2, 5120)
    np.testing.assert_allclose(pose_chunks[0, :YOLO_POSE_DIM], pose[:6].mean(axis=0))


def test_pose_pairwise_features_expand_base_vector_to_68() -> None:
    base = np.zeros((3, 57), dtype=np.float32)
    for point_idx in range(17):
        base[:, point_idx * 3] = point_idx
        base[:, point_idx * 3 + 1] = point_idx * 2
        base[:, point_idx * 3 + 2] = 1.0

    features = _add_pose_distances(base)

    assert features.shape == (3, YOLO_POSE_DIM)
    assert np.all(features[:, 57:] >= 0)


def test_rtmw_feature_vector_has_expected_layout() -> None:
    keypoints = np.zeros((133, 2), dtype=np.float32)
    scores = np.ones(133, dtype=np.float32)
    keypoints[:, 0] = 50
    keypoints[:, 1] = 40
    box = np.array([10, 20, 110, 220], dtype=np.float32)

    feature = _rtmw_feature(keypoints, scores, box, width=200, height=300)

    assert feature.shape == (RTMW_FRAME_DIM,)
    assert feature[200] == 1.0
    assert feature[201] == 1.0
    assert feature[207] == 1.0
    assert feature[208] == 1.0
    assert feature[209] == 1.0


def test_fusion_normalizer_is_built_from_current_video_chunks(tmp_path) -> None:
    pose_path = tmp_path / "pose.npz"
    rgb_path = tmp_path / "rgb.npz"
    output_path = tmp_path / "normalizers" / "video_fusion_normalizers.npz"
    pose = np.array([[1.0, 2.0], [3.0, 6.0]], dtype=np.float32)
    rgb = np.array([[10.0, 20.0], [14.0, 28.0]], dtype=np.float32)
    np.savez_compressed(pose_path, x=pose)
    np.savez_compressed(rgb_path, x=rgb)

    result = build_fusion_normalizer(pose_path, rgb_path, output_path)

    assert result == output_path
    data = np.load(output_path, allow_pickle=True)
    np.testing.assert_allclose(data["pose_mean"], [2.0, 4.0])
    np.testing.assert_allclose(data["pose_std"], [1.0, 2.0])
    np.testing.assert_allclose(data["rgb_mean"], [12.0, 24.0])
    np.testing.assert_allclose(data["rgb_std"], [2.0, 4.0])
    assert str(data["source"]) == "generated_from_uploaded_video"


def test_rtmw_with_no_selected_frames_writes_sparse_empty_artifact(tmp_path, monkeypatch) -> None:
    output_path = tmp_path / "rtmw" / "sample.npz"
    monkeypatch.setattr(
        "app.services.feature_extraction.get_video_meta",
        lambda path: {"width": 640, "height": 480, "n_frames": 120},
    )

    extract_rtmw(
        tmp_path / "sample.mp4",
        output_path,
        [],
        person_model_name="unused.pt",
        conf=0.3,
        imgsz=640,
    )

    data = np.load(output_path, allow_pickle=True)
    assert data["frame_indices"].shape == (0,)
    assert data["keypoints"].shape == (0, 133, 2)
    assert data["scores"].shape == (0, 133)
    assert data["boxes"].shape == (0, 4)
    assert str(data["source"]) == "RTMW WholeBody on action-selected frames"


def test_normalizer_path_stays_inside_repository_when_outputs_are_external(tmp_path, monkeypatch) -> None:
    repository = tmp_path / "repository"
    external_output = tmp_path / "external" / "job-42"
    video_path = tmp_path / "sample.mp4"

    def fake_pose(video, frame_path, chunk_path, **kwargs):
        chunk_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(chunk_path, x=np.ones((2, 272), dtype=np.float32))

    def fake_rgb(video, frame_path, chunk_path, **kwargs):
        chunk_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(chunk_path, x=np.ones((2, 5120), dtype=np.float32))

    monkeypatch.setattr("app.services.feature_extraction.REPO_ROOT", repository)
    monkeypatch.setattr("app.services.feature_extraction.extract_yolo_pose", fake_pose)
    monkeypatch.setattr("app.services.feature_extraction.extract_rgb", fake_rgb)

    from app.services.feature_extraction import extract_classification_features

    paths = extract_classification_features(video_path, external_output)

    expected = (
        repository
        / "data"
        / "outputs"
        / "job-42"
        / "features"
        / "normalizers"
        / "sample_fusion_normalizers.npz"
    )
    assert paths["normalizer"] == expected
    assert expected.exists()
