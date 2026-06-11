from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch
from tqdm import tqdm

from app.services.video import get_video_meta, video_id

CHUNK_SIZE = 6
YOLO_POSE_DIM = 68
RGB_FRAME_DIM = 1280
RTMW_FRAME_DIM = 213
ProgressCallback = Callable[[float, str, str], None]
REPO_ROOT = Path(__file__).resolve().parents[3]
BODY_IDS = [5, 6, 7, 8, 9, 10, 11, 12]
LEFT_HAND_IDS = list(range(91, 112))
RIGHT_HAND_IDS = list(range(112, 133))
SELECTED_RTM_IDS = BODY_IDS + LEFT_HAND_IDS + RIGHT_HAND_IDS
PAIRWISE_POINTS = [
    (5, 7),
    (7, 9),
    (5, 9),
    (6, 8),
    (8, 10),
    (6, 10),
    (9, 10),
    (5, 6),
    (11, 12),
    (5, 11),
    (6, 12),
]


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def processing_device() -> str:
    if torch.cuda.is_available():
        return f"CUDA: {torch.cuda.get_device_name(0)}"
    return "CPU"


def _report_progress(
    callback: ProgressCallback | None,
    fraction: float,
    stage: str,
    message: str,
) -> None:
    if callback is not None:
        callback(float(np.clip(fraction, 0.0, 1.0)), stage, message)


def _frame_progress(
    callback: ProgressCallback | None,
    frame_idx: int,
    total: int,
    stage: str,
) -> None:
    if total <= 0:
        return
    step = max(1, total // 100)
    current = frame_idx + 1
    if current == total or current % step == 0:
        _report_progress(callback, current / total, stage, f"Кадр {current} из {total}")


def _rtmw_device() -> str:
    try:
        import onnxruntime as ort

        if "CUDAExecutionProvider" in ort.get_available_providers():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _aggregate_chunks(features: np.ndarray, chunk_size: int = CHUNK_SIZE) -> np.ndarray:
    n_trim = len(features) // chunk_size * chunk_size
    if n_trim == 0:
        raise RuntimeError(f"Видео должно содержать не менее {chunk_size} кадров")
    chunks = features[:n_trim].reshape(-1, chunk_size, features.shape[1])
    mean = chunks.mean(axis=1)
    std = chunks.std(axis=1)
    dx = np.diff(chunks, axis=1)
    dx_mean = dx.mean(axis=1)
    dx_std = dx.std(axis=1)
    return np.concatenate([mean, std, dx_mean, dx_std], axis=1).astype(np.float32)


def _iter_video_frames(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def _select_main_pose(result) -> int | None:
    if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
        return None
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    conf = result.boxes.conf.detach().cpu().numpy()
    areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return int(np.argmax(areas * (conf + 1e-3)))


def _pose_base_feature(result, width: int, height: int) -> np.ndarray:
    feature = np.zeros(57, dtype=np.float32)
    feature[-1] = 1.0
    idx = _select_main_pose(result)
    if idx is None:
        return feature

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    conf = result.boxes.conf.detach().cpu().numpy()
    xy = result.keypoints.xy.detach().cpu().numpy()
    kpt_conf_tensor = result.keypoints.conf
    kpt_conf = (
        kpt_conf_tensor.detach().cpu().numpy()
        if kpt_conf_tensor is not None
        else np.ones(xy.shape[:2], dtype=np.float32)
    )
    x1, y1, x2, y2 = boxes[idx]
    bw = max(float(x2 - x1), 1.0)
    bh = max(float(y2 - y1), 1.0)
    scale = max(bw, bh)
    cx = float(x1 + x2) / 2.0
    cy = float(y1 + y2) / 2.0

    for point_idx in range(17):
        offset = point_idx * 3
        feature[offset] = (float(xy[idx, point_idx, 0]) - cx) / scale
        feature[offset + 1] = (float(xy[idx, point_idx, 1]) - cy) / scale
        feature[offset + 2] = float(kpt_conf[idx, point_idx])

    feature[51:57] = [cx / width, cy / height, bw / width, bh / height, float(conf[idx]), 0.0]
    return feature


def _smooth_pose_features(base: np.ndarray, max_gap: int = 5, window: int = 5) -> np.ndarray:
    import pandas as pd

    result = base.copy()
    coordinate_columns = [i for i in range(51) if i % 3 in (0, 1)] + [51, 52]
    for col in coordinate_columns:
        series = pd.Series(result[:, col])
        missing = result[:, 56] > 0.5
        series[missing] = np.nan
        series = series.interpolate(limit=max_gap, limit_direction="both")
        series = series.rolling(window=window, center=True, min_periods=1).median()
        result[:, col] = series.fillna(0.0).to_numpy(dtype=np.float32)
    return result


def _add_pose_distances(base: np.ndarray) -> np.ndarray:
    points = base[:, :51].reshape(len(base), 17, 3)
    distances = []
    for left, right in PAIRWISE_POINTS:
        dist = np.linalg.norm(points[:, left, :2] - points[:, right, :2], axis=1)
        confidence = np.minimum(points[:, left, 2], points[:, right, 2])
        distances.append((dist * (confidence > 0.05)).astype(np.float32))
    return np.concatenate([base, np.stack(distances, axis=1)], axis=1).astype(np.float32)


def extract_yolo_pose(
    video_path: Path,
    frame_path: Path,
    chunk_path: Path,
    model_name: str,
    imgsz: int,
    conf: float,
    progress_callback: ProgressCallback | None = None,
) -> None:
    from ultralytics import YOLO

    meta = get_video_meta(video_path)
    model = YOLO(model_name)
    rows = []
    for frame_idx, frame in enumerate(tqdm(_iter_video_frames(video_path), total=meta["n_frames"], desc="YOLOv8m-pose")):
        result = model.predict(frame, imgsz=imgsz, conf=conf, verbose=False, device=_device())[0]
        rows.append(_pose_base_feature(result, meta["width"], meta["height"]))
        _frame_progress(progress_callback, frame_idx, meta["n_frames"], "YOLOv8m-pose")

    base = np.asarray(rows, dtype=np.float32)
    smooth = _smooth_pose_features(base)
    features = _add_pose_distances(smooth)
    chunks = _aggregate_chunks(features)
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        frame_path,
        features=features,
        base_features=base,
        base_smooth=smooth,
        model_name=model_name,
        imgsz=imgsz,
        conf=conf,
        max_gap=5,
        smooth_window=5,
        video_id=video_id(video_path),
    )
    np.savez_compressed(
        chunk_path,
        x=chunks,
        y=np.zeros(len(chunks), dtype=np.int64),
        video_id=video_id(video_path),
        split="inference",
        chunk_size=CHUNK_SIZE,
        chunk_feature_type="mean_std_dxmean_dxstd",
    )


def extract_rgb(
    video_path: Path,
    frame_path: Path,
    chunk_path: Path,
    batch_size: int = 32,
    progress_callback: ProgressCallback | None = None,
) -> None:
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    device = torch.device(_device())
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights).to(device).eval()
    model.classifier = torch.nn.Identity()
    transform = weights.transforms()
    meta = get_video_meta(video_path)
    rows: list[np.ndarray] = []
    batch: list[torch.Tensor] = []

    def flush() -> None:
        if not batch:
            return
        tensor = torch.stack(batch).to(device)
        with torch.no_grad():
            output = model(tensor).detach().cpu().numpy().astype(np.float32)
        rows.extend(output)
        batch.clear()

    for frame_idx, frame in enumerate(tqdm(_iter_video_frames(video_path), total=meta["n_frames"], desc="EfficientNet-B0 RGB")):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)
        batch.append(transform(tensor))
        if len(batch) >= batch_size:
            flush()
        _frame_progress(progress_callback, frame_idx, meta["n_frames"], "EfficientNet-B0 RGB")
    flush()

    features = np.asarray(rows, dtype=np.float32)
    if features.shape[1] != RGB_FRAME_DIM:
        raise RuntimeError(f"Ожидалось {RGB_FRAME_DIM} RGB-признаков, получено {features.shape}")
    chunks = _aggregate_chunks(features)
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(frame_path, features=features, video_id=video_id(video_path), model_name="efficientnet_b0")
    np.savez_compressed(
        chunk_path,
        x=chunks,
        y=np.zeros(len(chunks), dtype=np.int64),
        video_id=video_id(video_path),
        split="inference",
        chunk_size=CHUNK_SIZE,
        chunk_feature_type="mean_std_dxmean_dxstd",
    )


def _select_person_box(result, frame_shape: tuple[int, ...], previous: np.ndarray | None, min_area_ratio: float) -> np.ndarray | None:
    if result.boxes is None or len(result.boxes) == 0:
        return None
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    conf = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    frame_area = float(frame_shape[0] * frame_shape[1])
    best_box = None
    best_score = -np.inf
    for box, score, cls_id in zip(boxes, conf, classes):
        if cls_id != 0:
            continue
        area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
        if area / frame_area < min_area_ratio:
            continue
        rank = area * (float(score) + 1e-3)
        if previous is not None:
            center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            prev_center = np.array([(previous[0] + previous[2]) / 2, (previous[1] + previous[3]) / 2])
            rank /= 1.0 + float(np.linalg.norm(center - prev_center))
        if rank > best_score:
            best_score = rank
            best_box = box.astype(np.float32)
    return best_box


def _expand_box(box: np.ndarray, width: int, height: int, factor: float = 1.20) -> np.ndarray:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    bw, bh = (box[2] - box[0]) * factor, (box[3] - box[1]) * factor
    return np.array([
        np.clip(cx - bw / 2, 0, width - 1),
        np.clip(cy - bh / 2, 0, height - 1),
        np.clip(cx + bw / 2, 0, width - 1),
        np.clip(cy + bh / 2, 0, height - 1),
    ], dtype=np.float32)


def _normalize_rtmw_output(keypoints: Any, scores: Any) -> tuple[np.ndarray, np.ndarray]:
    kpts = np.asarray(keypoints, dtype=np.float32)
    scs = np.asarray(scores, dtype=np.float32)
    if kpts.ndim == 2:
        kpts = kpts[None, ...]
    if scs.ndim == 1:
        scs = scs[None, ...]
    return kpts, scs


def _pose_quality(kpts: np.ndarray, scores: np.ndarray) -> float:
    body = scores[:17]
    valid = body >= 0.22
    if int(valid.sum()) < 3:
        return 0.0
    body_score = float(body[valid].mean())
    left = float(scores[91:112].mean()) if len(scores) >= 112 else 0.0
    right = float(scores[112:133].mean()) if len(scores) >= 133 else 0.0
    return 0.7 * body_score + 0.15 * left + 0.15 * right


def _rtmw_feature(kpts: np.ndarray, scores: np.ndarray, box: np.ndarray, width: int, height: int) -> np.ndarray:
    feature = np.zeros(RTMW_FRAME_DIM, dtype=np.float32)
    x1, y1, x2, y2 = box
    scale = max(float(x2 - x1), float(y2 - y1), 1.0)
    cx, cy = float(x1 + x2) / 2, float(y1 + y2) / 2
    offset = 0
    for point_id in SELECTED_RTM_IDS:
        score = float(scores[point_id]) if point_id < len(scores) else 0.0
        if point_id < len(kpts) and score >= 0.05 and np.isfinite(kpts[point_id]).all():
            feature[offset:offset + 4] = [
                (float(kpts[point_id, 0]) - cx) / scale,
                (float(kpts[point_id, 1]) - cy) / scale,
                score,
                1.0,
            ]
        offset += 4
    body_scores = scores[BODY_IDS]
    left_scores = scores[LEFT_HAND_IDS]
    right_scores = scores[RIGHT_HAND_IDS]
    feature[200:] = [
        1.0,
        1.0,
        x1 / width,
        y1 / height,
        x2 / width,
        y2 / height,
        ((x2 - x1) * (y2 - y1)) / (width * height),
        float((body_scores >= 0.05).mean()),
        float((left_scores >= 0.05).mean()),
        float((right_scores >= 0.05).mean()),
        float(body_scores.mean()),
        float(left_scores.mean()),
        float(right_scores.mean()),
    ]
    return feature


def extract_rtmw(
    video_path: Path,
    frame_path: Path,
    frame_indices: list[int] | np.ndarray,
    person_model_name: str,
    conf: float,
    imgsz: int,
    progress_callback: ProgressCallback | None = None,
) -> None:
    meta = get_video_meta(video_path)
    selected = np.asarray(sorted({
        int(idx) for idx in frame_indices if 0 <= int(idx) < meta["n_frames"]
    }), dtype=np.int64)
    selected_count = len(selected)
    keypoints_all = np.full((selected_count, 133, 2), np.nan, dtype=np.float32)
    scores_all = np.zeros((selected_count, 133), dtype=np.float32)
    boxes_all = np.zeros((selected_count, 4), dtype=np.float32)

    if selected_count == 0:
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            frame_path,
            boxes=boxes_all,
            video_id=video_id(video_path),
            split="inference",
            stats_json=json.dumps({
                "n_frames": meta["n_frames"],
                "selected_frames": 0,
                "processed_frames": 0,
            }, ensure_ascii=False),
            keypoints=keypoints_all,
            scores=scores_all,
            frame_indices=selected,
            source="RTMW WholeBody on action-selected frames",
        )
        _report_progress(progress_callback, 1.0, "RTMW WholeBody", "Кадры для локализации кистей не требуются")
        return

    from rtmlib import Wholebody
    from ultralytics import YOLO

    detector_device = _device()
    rtmw_device = _rtmw_device()
    detector = YOLO(person_model_name)
    try:
        wholebody = Wholebody(to_openpose=False, mode="performance", backend="onnxruntime", device=rtmw_device)
    except Exception:
        wholebody = Wholebody(to_openpose=False, backend="onnxruntime", device=rtmw_device)

    selected_positions = {int(frame_idx): pos for pos, frame_idx in enumerate(selected)}
    previous_box = None
    previous_keypoints = None
    previous_frame_idx = None
    processed = 0
    visited = 0

    for frame_idx, frame in enumerate(tqdm(_iter_video_frames(video_path), total=meta["n_frames"], desc="Selected frames: YOLO gate + RTMW")):
        position = selected_positions.get(frame_idx)
        if position is None:
            continue
        visited += 1
        if previous_frame_idx is None or frame_idx != previous_frame_idx + 1:
            previous_box = None
            previous_keypoints = None
        previous_frame_idx = frame_idx

        result = detector.predict(frame, imgsz=imgsz, conf=conf, classes=[0], verbose=False, device=detector_device)[0]
        box = _select_person_box(result, frame.shape, previous_box, min_area_ratio=0.010)
        if box is not None:
            box = _expand_box(box, meta["width"], meta["height"])
            if previous_box is not None:
                box = 0.65 * box + 0.35 * previous_box
            previous_box = box
            x1, y1, x2, y2 = box.astype(int)
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                kpts_batch, scores_batch = _normalize_rtmw_output(*wholebody(crop))
                if len(kpts_batch) > 0:
                    qualities = [_pose_quality(k, score) for k, score in zip(kpts_batch, scores_batch)]
                    best = int(np.argmax(qualities))
                    if qualities[best] >= 0.35:
                        kpts = kpts_batch[best].copy()
                        scores = scores_batch[best].copy()
                        kpts[:, 0] += x1
                        kpts[:, 1] += y1
                        if previous_keypoints is not None:
                            valid = np.isfinite(kpts).all(axis=1) & np.isfinite(previous_keypoints).all(axis=1)
                            kpts[valid] = 0.50 * kpts[valid] + 0.50 * previous_keypoints[valid]
                        previous_keypoints = kpts.copy()
                        count = min(133, len(kpts), len(scores))
                        keypoints_all[position, :count] = kpts[:count]
                        scores_all[position, :count] = scores[:count]
                        boxes_all[position] = box
                        processed += 1

        _frame_progress(progress_callback, visited - 1, selected_count, "YOLO + RTMW WholeBody")
        if visited == selected_count:
            break

    frame_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "n_frames": meta["n_frames"],
        "selected_frames": selected_count,
        "processed_frames": processed,
        "processed_rate": processed / max(1, selected_count),
    }
    np.savez_compressed(
        frame_path,
        boxes=boxes_all,
        video_id=video_id(video_path),
        split="inference",
        stats_json=json.dumps(stats, ensure_ascii=False),
        keypoints=keypoints_all,
        scores=scores_all,
        frame_indices=selected,
        source="RTMW WholeBody on action-selected frames",
    )


def build_fusion_normalizer(pose_chunk_path: Path, rgb_chunk_path: Path, output_path: Path) -> Path:
    pose_x = np.load(pose_chunk_path, allow_pickle=True)["x"].astype(np.float32)
    rgb_x = np.load(rgb_chunk_path, allow_pickle=True)["x"].astype(np.float32)
    if pose_x.ndim != 2 or rgb_x.ndim != 2 or len(pose_x) == 0 or len(rgb_x) == 0:
        raise RuntimeError("Недостаточно кадров для построения chunk-признаков и normalizer")
    if len(pose_x) != len(rgb_x):
        raise RuntimeError(
            f"Количество pose- и RGB-чанков не совпадает: {len(pose_x)} и {len(rgb_x)}"
        )
    pose_mean = pose_x.mean(axis=0).astype(np.float32)
    pose_std = np.maximum(pose_x.std(axis=0), 1e-6).astype(np.float32)
    rgb_mean = rgb_x.mean(axis=0).astype(np.float32)
    rgb_std = np.maximum(rgb_x.std(axis=0), 1e-6).astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        pose_mean=pose_mean,
        pose_std=pose_std,
        rgb_mean=rgb_mean,
        rgb_std=rgb_std,
        source="generated_from_uploaded_video",
    )
    return output_path


def extract_classification_features(
    video_path: Path,
    output_dir: Path,
    params: dict[str, Any] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Path]:
    params = params or {}
    vid = video_id(video_path)
    root = output_dir / "features"
    paths = {
        "pose_frame": root / "yolo_pose" / "frame_features" / f"{vid}.npz",
        "pose_chunk": root / "yolo_pose" / "chunk_features" / f"{vid}.npz",
        "rgb_frame": root / "rgb" / "frame_features" / f"{vid}.npz",
        "rgb_chunk": root / "rgb" / "chunk_features" / f"{vid}.npz",
        "rtmw_frame": root / "rtmw" / "frame_features" / f"{vid}.npz",
        "normalizer": (
            REPO_ROOT
            / "data"
            / "outputs"
            / output_dir.name
            / "features"
            / "normalizers"
            / f"{vid}_fusion_normalizers.npz"
        ),
    }
    force = bool(params.get("force_feature_extraction", False))

    def mapped(start: float, end: float, label: str) -> ProgressCallback:
        def callback(fraction: float, stage: str, message: str) -> None:
            _report_progress(
                progress_callback,
                start + (end - start) * fraction,
                stage,
                f"{label}: {message}",
            )
        return callback

    if force or not paths["pose_chunk"].exists():
        _report_progress(progress_callback, 0.0, "YOLOv8m-pose", "Загрузка модели позы")
        extract_yolo_pose(
            video_path,
            paths["pose_frame"],
            paths["pose_chunk"],
            model_name=str(params.get("yolo_pose_model", "yolov8m-pose.pt")),
            imgsz=int(params.get("yolo_imgsz", 640)),
            conf=float(params.get("yolo_pose_conf", 0.15)),
            progress_callback=mapped(0.0, 0.55, "Извлечение pose-признаков"),
        )
    else:
        _report_progress(progress_callback, 0.55, "YOLOv8m-pose", "Используются признаки текущего задания")

    if force or not paths["rgb_chunk"].exists():
        _report_progress(progress_callback, 0.55, "EfficientNet-B0", "Загрузка RGB-модели")
        extract_rgb(
            video_path,
            paths["rgb_frame"],
            paths["rgb_chunk"],
            batch_size=int(params.get("rgb_batch_size", 32)),
            progress_callback=mapped(0.55, 0.92, "Извлечение RGB-признаков"),
        )
    else:
        _report_progress(progress_callback, 0.92, "EfficientNet-B0", "Используются признаки текущего задания")

    _report_progress(progress_callback, 0.95, "Нормализация", "Расчет mean/std по извлеченным признакам")
    build_fusion_normalizer(paths["pose_chunk"], paths["rgb_chunk"], paths["normalizer"])
    normalizer_path = paths["normalizer"].resolve()
    print("Normalizer текущего видео:", normalizer_path)
    _report_progress(
        progress_callback,
        1.0,
        "Нормализация",
        f"Normalizer сохранен: {normalizer_path}",
    )
    return paths
