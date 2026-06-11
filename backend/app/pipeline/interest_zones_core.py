# Auto-exported from interest_zones_v5_heatmap_from_scratch.ipynb.
# Safe to import from FastAPI. Configure paths via environment variables.

# ============================================================
# Cell 1. Imports and global config
# ============================================================

import os
import gc
import json
import math
import time
import shutil
import warnings
import subprocess
from pathlib import Path
from collections import defaultdict, Counter

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.auto import tqdm

try:
    from IPython.display import display, Image, Video
except Exception:
    def display(*args, **kwargs):
        return None
    Image = None
    Video = None


def safe_display(obj):
    """Do not let notebook-only display calls break FastAPI processing."""
    try:
        display(obj)
    except Exception:
        pass


def safe_display_image(path):
    try:
        if Image is not None:
            safe_display(Image(filename=str(path)))
    except Exception:
        pass


def safe_display_video(path, width=900):
    try:
        if Video is not None:
            safe_display(Video(str(path), embed=True, width=width))
    except Exception:
        pass

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)

ACTION_NAMES = [
    "background",
    "reach_to_shelf",
    "retract_from_shelf",
    "hand_in_shelf",
    "inspect_product",
    "inspect_shelf",
]
ACTION_TO_ID = {name: i for i, name in enumerate(ACTION_NAMES)}

# Веса именно для пространственного интереса.
# inspect_product сильно занижен, потому что товар часто уже около тела, а не у полки.
ZONE_ACTION_WEIGHTS = {
    "background": 0.00,
    "reach_to_shelf": 1.35,
    "hand_in_shelf": 1.60,
    "retract_from_shelf": 1.05,
    "inspect_shelf": 0.35,
    "inspect_product": 0.05,
}

SHELF_DIRECTED_ACTIONS = [
    "reach_to_shelf",
    "hand_in_shelf",
    "retract_from_shelf",
    "inspect_shelf",
]


# ============================================================
# Cell 2. Local paths
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT_DIR = Path(os.getenv("RETAIL_ACTION_ROOT", str(REPO_ROOT)))

BEST_MODEL_PATH = Path(os.getenv(
    "RETAIL_BEST_MODEL_PATH",
    str(REPO_ROOT / "models" / "best_C6_LateFusion_Gated_BiGRU.pt"),
))

OUT_DIR = Path(os.getenv("RETAIL_OUT_DIR", str(ROOT_DIR / "outputs" / "interest_zones_v5_heatmap")))
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_DIR = OUT_DIR / "results"
IMAGE_DIR = OUT_DIR / "images"
VIDEO_OUT_DIR = OUT_DIR / "videos"
DEBUG_DIR = OUT_DIR / "debug"

for p in [RESULT_DIR, IMAGE_DIR, VIDEO_OUT_DIR, DEBUG_DIR]:
    p.mkdir(parents=True, exist_ok=True)

SHELF_ROI_POLYGON = None

# ============================================================
# Cell 3. General utilities
# ============================================================

def get_video_id(video_path):
    stem = Path(video_path).stem
    for suffix in ["_crop", "_video", "_processed"]:
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
    return stem


def get_video_meta(video_path):
    video_path = Path(video_path)
    assert video_path.exists(), f"Видео не найдено: {video_path}"

    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), f"Не удалось открыть видео: {video_path}"

    meta = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return meta


def safe_mkdir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def try_convert_to_h264(input_path, output_path=None):
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_name(input_path.stem + "_h264.mp4")
    output_path = Path(output_path)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("ffmpeg не найден. Оставляю raw mp4:", input_path)
        return input_path

    cmd = [
        ffmpeg,
        "-y",
        "-i", str(input_path),
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-preset", "fast",
        "-crf", "23",
        str(output_path),
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output_path
    except Exception as e:
        print("Ошибка ffmpeg, оставляю raw mp4:", e)
        return input_path


def read_frame(video_path, frame_idx=0):
    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), f"Не удалось открыть видео: {video_path}"
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    assert ok, f"Не удалось прочитать кадр {frame_idx}"
    return frame


# ============================================================
# Cell 4. Exact C6 architecture for checkpoint loading
# ============================================================

NUM_CLASSES = 6


class AttentionPooling(nn.Module):
    def __init__(self, input_dim, dropout=0.40):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, 1),
        )

    def forward(self, x):
        scores = self.attn(x).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return pooled


class GRUBranch(nn.Module):
    def __init__(self, input_dim, hidden_dim=96, num_layers=2, dropout=0.40):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout * 0.5),
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.pool = AttentionPooling(hidden_dim * 2, dropout=dropout)
        self.out_dim = hidden_dim * 2

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = self.input_proj(x)
        out, _ = self.gru(x)
        return self.pool(out)


class LateFusionGatedBiGRU(nn.Module):
    def __init__(self, pose_dim, rgb_dim, num_classes=6, hidden_dim=96, dropout=0.40):
        super().__init__()
        self.pose_branch = GRUBranch(pose_dim, hidden_dim=hidden_dim, num_layers=2, dropout=dropout)
        self.rgb_branch = GRUBranch(rgb_dim, hidden_dim=hidden_dim, num_layers=2, dropout=dropout)

        branch_out = hidden_dim * 2
        self.gate = nn.Sequential(
            nn.Linear(branch_out * 2, branch_out),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(branch_out, 2),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(branch_out),
            nn.Dropout(dropout),
            nn.Linear(branch_out, num_classes),
        )

    def forward(self, pose, rgb, return_gates=False):
        if pose.ndim == 2:
            pose = pose.unsqueeze(1)
        if rgb.ndim == 2:
            rgb = rgb.unsqueeze(1)

        pose_emb = self.pose_branch(pose)
        rgb_emb = self.rgb_branch(rgb)

        both = torch.cat([pose_emb, rgb_emb], dim=-1)
        gate_logits = self.gate(both)
        gates = torch.softmax(gate_logits, dim=-1)

        fused = gates[:, 0:1] * pose_emb + gates[:, 1:2] * rgb_emb
        logits = self.head(fused)

        if return_gates:
            return logits, gates
        return logits


def load_torch_checkpoint_safe(path, map_location=DEVICE):
    path = Path(path)
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "state_dict", "model", "net"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        return ckpt
    return ckpt


def load_c6_model(pose_dim, rgb_dim):
    assert BEST_MODEL_PATH.exists(), f"Не найден чекпоинт C6: {BEST_MODEL_PATH}"

    model = LateFusionGatedBiGRU(
        pose_dim=pose_dim,
        rgb_dim=rgb_dim,
        num_classes=NUM_CLASSES,
        hidden_dim=96,
        dropout=0.40,
    )

    ckpt = load_torch_checkpoint_safe(BEST_MODEL_PATH, map_location=DEVICE)
    state = extract_state_dict(ckpt)
    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()
    return model

print("C6 class is ready")


# ============================================================
# Cell 5. C6 action inference over chunks
# ============================================================

def load_c_normalizers(normalizer_path):
    normalizer_path = Path(normalizer_path)
    if not normalizer_path.exists():
        raise RuntimeError(f"Normalizer текущего видео не найден: {normalizer_path}")
    d = np.load(normalizer_path, allow_pickle=True)
    print("Normalizer:", normalizer_path)
    return {
        "pose_mean": d["pose_mean"].astype(np.float32),
        "pose_std": d["pose_std"].astype(np.float32),
        "rgb_mean": d["rgb_mean"].astype(np.float32),
        "rgb_std": d["rgb_std"].astype(np.float32),
    }


def load_expC_chunks(video_path, pose_chunk_path=None, rgb_chunk_path=None, normalizer_path=None):
    video_id = get_video_id(video_path)

    if pose_chunk_path is None or rgb_chunk_path is None or normalizer_path is None:
        raise RuntimeError("Pose, RGB и normalizer должны быть построены из загруженного видео перед классификацией")
    pose_path = Path(pose_chunk_path)
    rgb_path = Path(rgb_chunk_path)

    pose_npz = np.load(pose_path, allow_pickle=True)
    rgb_npz = np.load(rgb_path, allow_pickle=True)

    pose_x = pose_npz["x"].astype(np.float32)
    rgb_x = rgb_npz["x"].astype(np.float32)

    n = min(len(pose_x), len(rgb_x))
    pose_x = pose_x[:n]
    rgb_x = rgb_x[:n]

    norms = load_c_normalizers(normalizer_path)
    if pose_x.shape[1] != len(norms["pose_mean"]):
        raise RuntimeError(f"Размер pose-признаков {pose_x.shape[1]} не совпадает с normalizer {len(norms['pose_mean'])}")
    if rgb_x.shape[1] != len(norms["rgb_mean"]):
        raise RuntimeError(f"Размер RGB-признаков {rgb_x.shape[1]} не совпадает с normalizer {len(norms['rgb_mean'])}")
    pose_x = ((pose_x - norms["pose_mean"]) / np.maximum(norms["pose_std"], 1e-6)).astype(np.float32)
    rgb_x = ((rgb_x - norms["rgb_mean"]) / np.maximum(norms["rgb_std"], 1e-6)).astype(np.float32)

    # В текущих логах 10_1: 3796 кадров / 632 чанка ≈ 6 кадров.
    if "chunk_size" in pose_npz.files:
        chunk_size = int(np.asarray(pose_npz["chunk_size"]).item())
    elif "chunk_size" in rgb_npz.files:
        chunk_size = int(np.asarray(rgb_npz["chunk_size"]).item())
    else:
        meta = get_video_meta(video_path)
        chunk_size = max(1, int(round(meta["n_frames"] / n)))

    frame_ranges = [(i * chunk_size, min((i + 1) * chunk_size - 1, 10**12)) for i in range(n)]

    return {
        "video_id": video_id,
        "pose_x": pose_x,
        "rgb_x": rgb_x,
        "pose_path": pose_path,
        "rgb_path": rgb_path,
        "chunk_size": chunk_size,
        "frame_ranges": frame_ranges,
    }


def predict_actions_c6(
    video_path,
    batch_size=512,
    pose_chunk_path=None,
    rgb_chunk_path=None,
    normalizer_path=None,
):
    extracted = load_expC_chunks(
        video_path,
        pose_chunk_path=pose_chunk_path,
        rgb_chunk_path=rgb_chunk_path,
        normalizer_path=normalizer_path,
    )
    pose_x = extracted["pose_x"]
    rgb_x = extracted["rgb_x"]

    print("video_id:", extracted["video_id"])
    print("pose_x:", pose_x.shape)
    print("rgb_x:", rgb_x.shape)
    print("pose_path:", extracted["pose_path"])
    print("rgb_path:", extracted["rgb_path"])
    print("chunk_size:", extracted["chunk_size"])

    model = load_c6_model(pose_dim=pose_x.shape[1], rgb_dim=rgb_x.shape[1])

    probs_all = []
    preds_all = []
    confs_all = []

    for start in range(0, len(pose_x), batch_size):
        end = min(start + batch_size, len(pose_x))
        pose_batch = torch.tensor(pose_x[start:end], dtype=torch.float32, device=DEVICE)
        rgb_batch = torch.tensor(rgb_x[start:end], dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            logits = model(pose_batch, rgb_batch)
            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)

        probs_all.append(probs.cpu().numpy())
        preds_all.append(pred.cpu().numpy())
        confs_all.append(conf.cpu().numpy())

    probs = np.concatenate(probs_all, axis=0)
    preds = np.concatenate(preds_all, axis=0)
    confs = np.concatenate(confs_all, axis=0)

    rows = []
    for i, (fs, fe) in enumerate(extracted["frame_ranges"]):
        pred_id = int(preds[i])
        pred_name = ACTION_NAMES[pred_id]

        row = {
            "chunk_idx": i,
            "frame_start": int(fs),
            "frame_end": int(fe),
            "pred_class_id": pred_id,
            "pred_class_name": pred_name,
            "confidence": float(confs[i]),
        }

        weighted_signal = 0.0
        weighted_parts = {}
        for c_id, c_name in enumerate(ACTION_NAMES):
            p = float(probs[i, c_id])
            row[f"prob_{c_name}"] = p
            part = p * float(ZONE_ACTION_WEIGHTS.get(c_name, 0.0))
            weighted_parts[c_name] = part
            weighted_signal += part

        best_zone_action = max(weighted_parts, key=weighted_parts.get)
        row["zone_action_signal"] = float(weighted_signal)
        row["zone_top_action"] = best_zone_action
        row["zone_top_action_score"] = float(weighted_parts[best_zone_action])
        rows.append(row)

    pred_df = pd.DataFrame(rows)
    return pred_df


def get_prediction_for_frame(pred_df, frame_idx):
    g = pred_df[(pred_df["frame_start"] <= frame_idx) & (pred_df["frame_end"] >= frame_idx)]
    if len(g) > 0:
        return g.iloc[0]

    # fallback на ближайший чанк
    centers = (pred_df["frame_start"].to_numpy() + pred_df["frame_end"].to_numpy()) / 2.0
    idx = int(np.argmin(np.abs(centers - frame_idx)))
    return pred_df.iloc[idx]


# ============================================================
# Cell 6. RTMW WholeBody keypoint loader
# ============================================================

COCO_BODY = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

# COCO-WholeBody: 0-16 body, 17-22 foot, 23-90 face, 91-111 left hand, 112-132 right hand.
LEFT_HAND_START = 91
RIGHT_HAND_START = 112
FINGERTIP_LOCAL_IDS = [4, 8, 12, 16, 20]


def find_keypoints_array(npz):
    candidates = [
        "keypoints",
        "kpts",
        "points",
        "keypoints_xy",
        "pred_keypoints",
        "rtmw_keypoints",
    ]
    for name in candidates:
        if name in npz.files:
            arr = npz[name]
            if arr.ndim == 3 and arr.shape[-1] >= 2:
                return name, arr
    return None, None


def find_scores_array(npz, keypoints_arr=None):
    candidates = [
        "scores",
        "keypoint_scores",
        "kpt_scores",
        "keypoints_score",
        "keypoints_scores",
        "pred_scores",
        "rtmw_scores",
    ]
    for name in candidates:
        if name in npz.files:
            arr = npz[name]
            if arr.ndim == 2:
                return name, arr

    if keypoints_arr is not None and keypoints_arr.shape[-1] >= 3:
        return "keypoints[...,2]", keypoints_arr[..., 2]

    return None, None


def normalize_kpt_score(score):
    score = float(score)
    if not np.isfinite(score):
        return 0.0
    if score <= 1.0:
        return float(np.clip(score, 0.0, 1.0))
    # Для RTMW scores из текущего ноутбука встречались значения около 4-5.
    return float(score / (score + 1.0))


def safe_point(kpts, scores, idx, raw_score_thr=0.05):
    if idx >= len(kpts):
        return None

    x, y = kpts[idx]
    raw_score = float(scores[idx])

    if not np.isfinite(x) or not np.isfinite(y):
        return None
    if not np.isfinite(raw_score) or raw_score < raw_score_thr:
        return None

    return np.array([float(x), float(y)], dtype=np.float32), raw_score, normalize_kpt_score(raw_score)


def mean_valid_points(kpts, scores, ids, raw_score_thr=0.05):
    pts = []
    confs = []
    raws = []

    for idx in ids:
        p = safe_point(kpts, scores, idx, raw_score_thr=raw_score_thr)
        if p is None:
            continue
        xy, raw_score, conf = p
        pts.append(xy)
        confs.append(conf)
        raws.append(raw_score)

    if len(pts) == 0:
        return None

    pts = np.asarray(pts, dtype=np.float32)
    confs = np.asarray(confs, dtype=np.float32)
    weights = confs / (confs.sum() + 1e-6)
    xy = (pts * weights[:, None]).sum(axis=0)
    return xy, float(np.mean(raws)), float(np.mean(confs))


def get_body_anchor(kpts, scores):
    ids = [
        COCO_BODY["left_shoulder"],
        COCO_BODY["right_shoulder"],
        COCO_BODY["left_hip"],
        COCO_BODY["right_hip"],
    ]
    res = mean_valid_points(kpts, scores, ids, raw_score_thr=0.05)
    if res is None:
        return None
    return res[0]


def get_body_scale(kpts, scores, fallback=300.0):
    body_ids = [
        COCO_BODY["left_shoulder"],
        COCO_BODY["right_shoulder"],
        COCO_BODY["left_hip"],
        COCO_BODY["right_hip"],
        COCO_BODY["left_elbow"],
        COCO_BODY["right_elbow"],
        COCO_BODY["left_wrist"],
        COCO_BODY["right_wrist"],
    ]

    pts = []
    for idx in body_ids:
        p = safe_point(kpts, scores, idx, raw_score_thr=0.05)
        if p is not None:
            pts.append(p[0])

    if len(pts) < 2:
        return fallback

    pts = np.asarray(pts, dtype=np.float32)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    return max(float(np.hypot(x2 - x1, y2 - y1)), 80.0)


def get_hand_candidate(kpts, scores, side, raw_score_thr=0.05):
    if side == "left":
        start = LEFT_HAND_START
        wrist_idx = COCO_BODY["left_wrist"]
    else:
        start = RIGHT_HAND_START
        wrist_idx = COCO_BODY["right_wrist"]

    fingertip_ids = [start + i for i in FINGERTIP_LOCAL_IDS]
    hand_ids = [start + i for i in range(21)]

    # Лучший источник для зоны — fingertips.
    res = mean_valid_points(kpts, scores, fingertip_ids, raw_score_thr=raw_score_thr)
    if res is not None:
        xy, raw_score, conf = res
        return {"xy": xy, "raw_score": raw_score, "conf": conf, "source": f"{side}_fingertips"}

    # Запасной вариант — вся кисть.
    res = mean_valid_points(kpts, scores, hand_ids, raw_score_thr=raw_score_thr)
    if res is not None:
        xy, raw_score, conf = res
        return {"xy": xy, "raw_score": raw_score, "conf": conf, "source": f"{side}_hand_mean"}

    # Последний fallback — wrist.
    res = safe_point(kpts, scores, wrist_idx, raw_score_thr=raw_score_thr)
    if res is not None:
        xy, raw_score, conf = res
        return {"xy": xy, "raw_score": raw_score, "conf": conf, "source": f"{side}_wrist_fallback"}

    return None


# ============================================================
# Cell 7. Shelf ROI and geometry helpers
# ============================================================

def polygon_to_np(poly):
    arr = np.asarray(poly, dtype=np.float32)
    assert arr.ndim == 2 and arr.shape[1] == 2, "ROI polygon должен иметь форму [N, 2]"
    return arr


def make_roi_mask(width, height, polygon):
    mask = np.zeros((height, width), dtype=np.uint8)
    poly_i = polygon_to_np(polygon).astype(np.int32)
    cv2.fillPoly(mask, [poly_i], 255)
    return mask


def point_in_polygon(xy, polygon):
    poly = polygon_to_np(polygon)
    return cv2.pointPolygonTest(poly, (float(xy[0]), float(xy[1])), False) >= 0


def nearest_point_on_segment(p, a, b):
    p = np.asarray(p, dtype=np.float32)
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-6:
        return a.copy(), float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    q = a + t * ab
    return q, float(np.linalg.norm(p - q))


def nearest_point_on_polygon(p, polygon):
    poly = polygon_to_np(polygon)
    best_q = None
    best_d = float("inf")
    for i in range(len(poly)):
        a = poly[i]
        b = poly[(i + 1) % len(poly)]
        q, d = nearest_point_on_segment(p, a, b)
        if d < best_d:
            best_q = q
            best_d = d
    return best_q, best_d


def auto_shelf_roi_from_candidates(candidates_df, width, height, min_points=20, margin_ratio=0.14):
    """
    Авто-ROI не заменяет ручной shelf polygon.
    Это стартовая оценка по облаку RTMW hand candidates, чтобы быстро получить первую тепловую карту.
    """
    if candidates_df is None or len(candidates_df) < min_points:
        return np.array(
            [
                [0.15 * width, 0.05 * height],
                [0.98 * width, 0.05 * height],
                [0.98 * width, 0.90 * height],
                [0.15 * width, 0.90 * height],
            ],
            dtype=np.float32,
        )

    df = candidates_df.copy()
    if "action_signal" in df.columns:
        df = df[df["action_signal"] >= max(0.10, df["action_signal"].quantile(0.35))]
    if "extension_norm" in df.columns:
        df = df[df["extension_norm"] >= max(0.25, df["extension_norm"].quantile(0.25))]

    if len(df) < min_points:
        df = candidates_df.copy()

    x1, x2 = df["x"].quantile([0.03, 0.97]).to_numpy()
    y1, y2 = df["y"].quantile([0.03, 0.97]).to_numpy()

    mx = (x2 - x1) * margin_ratio + 40
    my = (y2 - y1) * margin_ratio + 40

    x1 = float(np.clip(x1 - mx, 0, width - 1))
    x2 = float(np.clip(x2 + mx, 0, width - 1))
    y1 = float(np.clip(y1 - my, 0, height - 1))
    y2 = float(np.clip(y2 + my, 0, height - 1))

    # Не даём ROI схлопнуться.
    if (x2 - x1) < width * 0.25:
        cx = (x1 + x2) / 2
        x1 = max(0, cx - width * 0.15)
        x2 = min(width - 1, cx + width * 0.15)
    if (y2 - y1) < height * 0.25:
        cy = (y1 + y2) / 2
        y1 = max(0, cy - height * 0.15)
        y2 = min(height - 1, cy + height * 0.15)

    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def get_shelf_roi(meta, candidates_df=None, manual_polygon=None):
    width, height = meta["width"], meta["height"]
    if manual_polygon is not None:
        polygon = polygon_to_np(manual_polygon)
        source = "manual"
    else:
        polygon = auto_shelf_roi_from_candidates(candidates_df, width, height)
        source = "auto_from_candidates"
    return polygon, source


def project_hand_to_shelf(hand_xy, torso_xy, roi_polygon, projection_alpha=0.45, max_projection_alpha=0.90, snap_distance=45):
    """
    Переносит точку интереса с самой кисти чуть дальше по направлению torso -> hand.
    Это помогает попасть ближе к товару/полке, а не к запястью перед телом.
    """
    hand_xy = np.asarray(hand_xy, dtype=np.float32)
    torso_xy = np.asarray(torso_xy, dtype=np.float32)
    vec = hand_xy - torso_xy
    norm = float(np.linalg.norm(vec))

    if not np.isfinite(norm) or norm < 1e-6:
        return None, "bad_direction"

    # Если сама кисть уже внутри shelf ROI, берём не wrist, а дальнюю точку на луче внутри ROI.
    samples = []
    for alpha in np.linspace(0.0, max_projection_alpha, 31):
        p = hand_xy + alpha * vec
        if point_in_polygon(p, roi_polygon):
            samples.append((alpha, p))

    if samples:
        # Берём самую дальнюю точку внутри ROI, но не дальше разумной projection_alpha, если такие есть.
        preferred = [item for item in samples if item[0] <= projection_alpha + 1e-6]
        if preferred:
            alpha, p = preferred[-1]
        else:
            alpha, p = samples[-1]
        return p.astype(np.float32), f"projected_alpha_{alpha:.2f}"

    # Если луч не попал в ROI, но кисть очень рядом с ROI, можно мягко притянуть к границе.
    nearest, dist = nearest_point_on_polygon(hand_xy, roi_polygon)
    if dist <= snap_distance:
        return nearest.astype(np.float32), f"snapped_to_roi_{dist:.1f}px"

    return None, "outside_roi"


# ============================================================
# Cell 8. Build raw RTMW hand candidates with C6 action signal
# ============================================================

def build_raw_hand_candidates(
    video_path,
    pred_df,
    raw_score_thr=0.05,
    min_action_signal=0.05,
    min_extension=0.18,
    rtmw_frame_path=None,
):
    video_id = get_video_id(video_path)
    if rtmw_frame_path:
        d = np.load(Path(rtmw_frame_path), allow_pickle=True)
        k_name, keypoints = find_keypoints_array(d)
        if keypoints is None:
            raise RuntimeError("В RTMW npz не найдены keypoints")
        s_name, scores = find_scores_array(d, keypoints)
        raw = {
            "path": Path(rtmw_frame_path),
            "keypoints_xy": keypoints[..., :2].astype(np.float32),
            "scores": scores.astype(np.float32) if scores is not None else np.ones(keypoints.shape[:2], dtype=np.float32),
            "frame_indices": d["frame_indices"].astype(np.int64) if "frame_indices" in d.files else np.arange(keypoints.shape[0]),
            "keypoints_key": k_name,
            "scores_key": s_name,
        }
    else:
        raise RuntimeError("RTMW-признаки должны быть извлечены из загруженного видео перед анализом")

    keypoints_xy = raw["keypoints_xy"]
    scores = raw["scores"]
    frame_indices = raw["frame_indices"]

    rows = []

    for t in tqdm(range(len(frame_indices)), desc="RTMW raw hand candidates"):
        frame_idx = int(frame_indices[t])
        kpts = keypoints_xy[t]
        scs = scores[t]

        pred_row = get_prediction_for_frame(pred_df, frame_idx)
        action_signal = float(pred_row["zone_action_signal"])
        if action_signal < min_action_signal:
            continue

        torso = get_body_anchor(kpts, scs)
        if torso is None:
            continue

        body_scale = get_body_scale(kpts, scs)

        for side in ["left", "right"]:
            cand = get_hand_candidate(kpts, scs, side=side, raw_score_thr=raw_score_thr)
            if cand is None:
                continue

            xy = cand["xy"]
            dist_to_torso = float(np.linalg.norm(xy - torso))
            extension_norm = dist_to_torso / (body_scale + 1e-6)
            if extension_norm < min_extension:
                continue

            # Действие для подписи зоны: максимум среди взвешенных action probabilities.
            weighted_parts = {}
            for action_name in ACTION_NAMES:
                prob_col = f"prob_{action_name}"
                p = float(pred_row[prob_col]) if prob_col in pred_row.index else 0.0
                weighted_parts[action_name] = p * float(ZONE_ACTION_WEIGHTS.get(action_name, 0.0))
            zone_top_action = max(weighted_parts, key=weighted_parts.get)

            rows.append({
                "frame_idx": frame_idx,
                "hand": side,
                "x": float(xy[0]),
                "y": float(xy[1]),
                "torso_x": float(torso[0]),
                "torso_y": float(torso[1]),
                "point_source": cand["source"],
                "hand_raw_score": float(cand["raw_score"]),
                "hand_conf": float(cand["conf"]),
                "body_scale": float(body_scale),
                "dist_to_torso": float(dist_to_torso),
                "extension_norm": float(extension_norm),
                "pred_class_name": str(pred_row["pred_class_name"]),
                "confidence": float(pred_row["confidence"]),
                "action_signal": action_signal,
                "zone_top_action": zone_top_action,
                "zone_top_action_score": float(weighted_parts[zone_top_action]),
                **{f"prob_{a}": float(pred_row[f"prob_{a}"]) for a in ACTION_NAMES if f"prob_{a}" in pred_row.index},
            })

    df = pd.DataFrame(rows)
    return df


def summarize_candidates(candidates_df):
    print("Candidates:", len(candidates_df))
    if len(candidates_df) == 0:
        return
    print("x range:", float(candidates_df["x"].min()), float(candidates_df["x"].max()))
    print("y range:", float(candidates_df["y"].min()), float(candidates_df["y"].max()))
    print("extension range:", float(candidates_df["extension_norm"].min()), float(candidates_df["extension_norm"].max()))
    display(candidates_df["zone_top_action"].value_counts())
    display(candidates_df.head())


# ============================================================
# Cell 9. Event-based shelf contact points
# ============================================================

def build_event_points_from_candidates(
    candidates_df,
    meta,
    roi_polygon,
    min_extension=0.32,
    min_action_signal=0.12,
    min_event_score=0.015,
    max_gap_frames=18,
    spatial_jump_px=130,
    peak_top_k=1,
    projection_alpha=0.45,
    max_projection_alpha=0.90,
    snap_distance=45,
):
    if candidates_df is None or len(candidates_df) == 0:
        return pd.DataFrame()

    df = candidates_df.copy()

    # Оставляем только shelf-directed смысл.
    df = df[df["zone_top_action"].isin(SHELF_DIRECTED_ACTIONS)].copy()
    df = df[df["extension_norm"] >= min_extension].copy()
    df = df[df["action_signal"] >= min_action_signal].copy()

    if len(df) == 0:
        return pd.DataFrame()

    projected_rows = []

    for row in df.itertuples(index=False):
        hand_xy = np.array([row.x, row.y], dtype=np.float32)
        torso_xy = np.array([row.torso_x, row.torso_y], dtype=np.float32)

        p, mode = project_hand_to_shelf(
            hand_xy=hand_xy,
            torso_xy=torso_xy,
            roi_polygon=roi_polygon,
            projection_alpha=projection_alpha,
            max_projection_alpha=max_projection_alpha,
            snap_distance=snap_distance,
        )

        if p is None:
            continue

        d = row._asdict()
        d["raw_x"] = float(d.pop("x"))
        d["raw_y"] = float(d.pop("y"))
        d["x"] = float(np.clip(p[0], 0, meta["width"] - 1))
        d["y"] = float(np.clip(p[1], 0, meta["height"] - 1))
        d["projection_mode"] = mode

        # Нелинейный вес: важны действие, вынос руки, уверенность hand points.
        extension_gate = float(np.clip((d["extension_norm"] - min_extension) / 0.45, 0.0, 1.0))
        d["interest_score"] = float(
            d["action_signal"] *
            (0.25 + 0.75 * extension_gate) *
            d["hand_conf"] *
            (d["extension_norm"] ** 1.35)
        )
        projected_rows.append(d)

    proj = pd.DataFrame(projected_rows)
    if len(proj) == 0:
        return proj

    proj = proj[proj["interest_score"] >= min_event_score].copy()
    if len(proj) == 0:
        return proj

    # Группируем в события отдельно по руке, но не по hard action.
    # Иначе один реальный reach/retract может дробиться из-за скачка предсказанного класса.
    selected = []

    for hand, g in proj.groupby("hand"):
        g = g.sort_values("frame_idx").reset_index(drop=True)

        event_id = 0
        event_ids = []
        prev_frame = None
        prev_xy = None

        for r in g.itertuples(index=False):
            xy = np.array([r.x, r.y], dtype=np.float32)
            new_event = False
            if prev_frame is None:
                new_event = True
            elif int(r.frame_idx) - int(prev_frame) > max_gap_frames:
                new_event = True
            elif prev_xy is not None and float(np.linalg.norm(xy - prev_xy)) > spatial_jump_px:
                new_event = True

            if new_event:
                event_id += 1

            event_ids.append(f"{hand}_{event_id:04d}")
            prev_frame = int(r.frame_idx)
            prev_xy = xy

        g["event_id"] = event_ids

        for ev_id, ev in g.groupby("event_id"):
            ev = ev.copy()
            ev["event_rank_score"] = (
                ev["interest_score"] *
                (1.0 + ev["extension_norm"]) *
                (1.0 + 0.25 * ev["zone_top_action"].eq("hand_in_shelf").astype(float))
            )
            top = ev.sort_values("event_rank_score", ascending=False).head(peak_top_k).copy()
            top["event_size_frames"] = int(ev["frame_idx"].nunique())
            selected.append(top)

    if not selected:
        return pd.DataFrame()

    events = pd.concat(selected, axis=0).sort_values("frame_idx").reset_index(drop=True)
    events["event_point_id"] = np.arange(len(events))
    return events


# ============================================================
# Cell 10. Heatmap and heatmap-zone extraction
# ============================================================

def add_gaussian_blob(heatmap, x, y, value=1.0, radius=40):
    h, w = heatmap.shape[:2]
    x = int(round(x))
    y = int(round(y))
    radius = int(max(3, radius))

    x1 = max(0, x - radius)
    x2 = min(w, x + radius + 1)
    y1 = max(0, y - radius)
    y2 = min(h, y + radius + 1)
    if x1 >= x2 or y1 >= y2:
        return

    xs = np.arange(x1, x2, dtype=np.float32)
    ys = np.arange(y1, y2, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    sigma = radius / 2.2
    blob = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    heatmap[y1:y2, x1:x2] += value * blob.astype(np.float32)


def build_interest_heatmap(event_points_df, meta, roi_polygon, radius_px=None, blur_px=0):
    width, height = meta["width"], meta["height"]
    heatmap = np.zeros((height, width), dtype=np.float32)

    if radius_px is None:
        radius_px = int(round(0.055 * min(width, height)))

    if event_points_df is not None and len(event_points_df) > 0:
        for r in event_points_df.itertuples(index=False):
            add_gaussian_blob(
                heatmap,
                x=float(r.x),
                y=float(r.y),
                value=float(r.interest_score),
                radius=radius_px,
            )

    roi_mask = make_roi_mask(width, height, roi_polygon)
    heatmap[roi_mask == 0] = 0.0

    if blur_px and blur_px > 0:
        k = int(blur_px)
        if k % 2 == 0:
            k += 1
        heatmap = cv2.GaussianBlur(heatmap, (k, k), 0)
        heatmap[roi_mask == 0] = 0.0

    return heatmap, roi_mask


def normalize_heatmap_0_255(heatmap):
    if heatmap.max() <= 1e-9:
        return np.zeros_like(heatmap, dtype=np.uint8)
    hm = heatmap / heatmap.max()
    return np.clip(hm * 255, 0, 255).astype(np.uint8)


def threshold_heatmap_mask(
    heatmap,
    roi_mask,
    relative_thr=0.24,
    percentile_thr=76,
    min_positive_value=1e-8,
    merge_dilate_px=24,
    open_px=5,
):
    mask_roi = roi_mask > 0
    vals = heatmap[mask_roi & (heatmap > min_positive_value)]
    if len(vals) == 0:
        return np.zeros_like(roi_mask, dtype=np.uint8), 0.0

    thr_rel = float(heatmap.max() * relative_thr)
    thr_pct = float(np.percentile(vals, percentile_thr))
    thr = max(thr_rel, thr_pct)

    binary = ((heatmap >= thr) & mask_roi).astype(np.uint8) * 255

    if open_px and open_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)

    if merge_dilate_px and merge_dilate_px > 0:
        ksize = int(merge_dilate_px)
        if ksize % 2 == 0:
            ksize += 1
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        binary = cv2.dilate(binary, k, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)

    binary[roi_mask == 0] = 0
    return binary, thr


def point_mask_values(mask, xs, ys):
    h, w = mask.shape[:2]
    xs = np.clip(np.round(xs).astype(int), 0, w - 1)
    ys = np.clip(np.round(ys).astype(int), 0, h - 1)
    return mask[ys, xs]


def extract_heatmap_zones(
    heatmap,
    roi_mask,
    event_points_df,
    min_component_area=350,
    relative_thr=0.24,
    percentile_thr=76,
    merge_dilate_px=24,
):
    binary, used_thr = threshold_heatmap_mask(
        heatmap,
        roi_mask,
        relative_thr=relative_thr,
        percentile_thr=percentile_thr,
        merge_dilate_px=merge_dilate_px,
    )

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    rows = []
    zone_masks = {}

    for label_id in range(1, num_labels):
        x, y, w, h, area = stats[label_id]
        if area < min_component_area:
            continue

        comp_mask = (labels == label_id).astype(np.uint8)
        raw_interest = float(heatmap[comp_mask > 0].sum())
        peak_interest = float(heatmap[comp_mask > 0].max()) if area > 0 else 0.0

        zone_points = pd.DataFrame()
        if event_points_df is not None and len(event_points_df) > 0:
            vals = point_mask_values(
                comp_mask,
                event_points_df["x"].to_numpy(),
                event_points_df["y"].to_numpy(),
            )
            zone_points = event_points_df[vals > 0].copy()

        if len(zone_points) > 0:
            top_action_scores = zone_points.groupby("zone_top_action")["interest_score"].sum().sort_values(ascending=False)
            top_action = str(top_action_scores.index[0])
            top_action_score = float(top_action_scores.iloc[0])
            n_points = int(len(zone_points))
            active_frames = int(zone_points["frame_idx"].nunique())
            point_score_sum = float(zone_points["interest_score"].sum())
        else:
            top_action = np.nan
            top_action_score = 0.0
            n_points = 0
            active_frames = 0
            point_score_sum = 0.0

        rows.append({
            "zone_id": f"HZ_{len(rows) + 1:02d}",
            "label_id": int(label_id),
            "x1": int(x),
            "y1": int(y),
            "x2": int(x + w),
            "y2": int(y + h),
            "center_x": float(centroids[label_id][0]),
            "center_y": float(centroids[label_id][1]),
            "component_area_px": int(area),
            "raw_interest": raw_interest,
            "peak_interest": peak_interest,
            "point_score_sum": point_score_sum,
            "n_event_points": n_points,
            "active_frames": active_frames,
            "top_action": top_action,
            "top_action_score": top_action_score,
        })
        zone_masks[label_id] = comp_mask

    zones_df = pd.DataFrame(rows)
    if len(zones_df) == 0:
        return zones_df, binary, used_thr, labels, zone_masks

    zones_df = zones_df.sort_values(
        ["raw_interest", "point_score_sum", "n_event_points"],
        ascending=False,
    ).reset_index(drop=True)
    zones_df["rank"] = np.arange(1, len(zones_df) + 1)
    zones_df["zone_id"] = [f"HZ_{i:02d}" for i in zones_df["rank"]]

    max_interest = zones_df["raw_interest"].max()
    zones_df["interest_0_100"] = zones_df["raw_interest"] / max_interest * 100.0 if max_interest > 0 else 0.0

    return zones_df, binary, used_thr, labels, zone_masks


# ============================================================
# Cell 11. Visualization utilities
# ============================================================

def heatmap_overlay_on_frame(frame_bgr, heatmap, alpha=0.48):
    hm_u8 = normalize_heatmap_0_255(heatmap)
    colored = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    mask = hm_u8 > 0
    out = frame_bgr.copy()
    out[mask] = cv2.addWeighted(frame_bgr, 1 - alpha, colored, alpha, 0)[mask]
    return out


def draw_roi(frame, roi_polygon, color=(255, 255, 0), thickness=2):
    poly = polygon_to_np(roi_polygon).astype(np.int32)
    cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=thickness)
    cv2.putText(
        frame,
        "shelf ROI",
        tuple(poly[0]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_heatmap_zones(frame, zones_df, max_zones=8):
    if zones_df is None or len(zones_df) == 0:
        return frame

    for _, z in zones_df.head(max_zones).iterrows():
        x1, y1, x2, y2 = int(z.x1), int(z.y1), int(z.x2), int(z.y2)
        rank = int(z["rank"])
        score = float(z["interest_0_100"])
        label = f"#{rank} {z.zone_id} {score:.0f}"
        if isinstance(z.top_action, str):
            label += f" {z.top_action}"

        color = (0, 255, 255) if rank == 1 else (0, 200, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(22, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
            cv2.LINE_AA,
        )
    return frame


def draw_event_points(frame, event_points_df, current_frame=None, temporal_window=5, max_points=250):
    if event_points_df is None or len(event_points_df) == 0:
        return frame

    df = event_points_df
    if current_frame is not None:
        df = df[np.abs(df["frame_idx"] - int(current_frame)) <= temporal_window]
    else:
        df = df.sort_values("interest_score", ascending=False).head(max_points)

    for r in df.itertuples(index=False):
        x, y = int(round(r.x)), int(round(r.y))
        rx, ry = int(round(r.raw_x)), int(round(r.raw_y)) if "raw_y" in df.columns else (x, y)
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        if "raw_x" in df.columns and "raw_y" in df.columns:
            cv2.circle(frame, (int(round(r.raw_x)), int(round(r.raw_y))), 3, (255, 255, 255), -1)
            cv2.line(frame, (int(round(r.raw_x)), int(round(r.raw_y))), (x, y), (255, 255, 255), 1)
    return frame


def save_summary_image(video_path, heatmap, roi_polygon, zones_df, event_points_df, out_path, frame_idx=0):
    frame = read_frame(video_path, frame_idx=frame_idx)
    canvas = heatmap_overlay_on_frame(frame, heatmap, alpha=0.50)
    draw_roi(canvas, roi_polygon)
    draw_heatmap_zones(canvas, zones_df)
    draw_event_points(canvas, event_points_df, current_frame=None, max_points=300)

    out_path = Path(out_path)
    safe_mkdir(out_path.parent)
    cv2.imwrite(str(out_path), canvas)
    return out_path


# ============================================================
# Cell 12. Render annotated video
# ============================================================

def render_interest_video(
    video_path,
    heatmap,
    roi_polygon,
    zones_df,
    event_points_df,
    out_path,
    max_frames=None,
    heatmap_alpha=0.35,
    temporal_window=5,
):
    video_path = Path(video_path)
    out_path = Path(out_path)
    safe_mkdir(out_path.parent)

    meta = get_video_meta(video_path)
    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), f"Не удалось открыть видео: {video_path}"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(out_path),
        fourcc,
        meta["fps"] if meta["fps"] > 0 else 30.0,
        (meta["width"], meta["height"]),
    )

    if max_frames is None:
        max_frames = meta["n_frames"]
    else:
        max_frames = min(int(max_frames), meta["n_frames"])

    for frame_idx in tqdm(range(max_frames), desc="Render heatmap interest video"):
        ok, frame = cap.read()
        if not ok:
            break

        canvas = heatmap_overlay_on_frame(frame, heatmap, alpha=heatmap_alpha)
        draw_roi(canvas, roi_polygon)
        draw_heatmap_zones(canvas, zones_df, max_zones=6)
        draw_event_points(canvas, event_points_df, current_frame=frame_idx, temporal_window=temporal_window)

        cv2.putText(
            canvas,
            f"frame {frame_idx}",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(canvas)

    cap.release()
    writer.release()

    converted = try_convert_to_h264(out_path)
    return converted


# ============================================================
# Cell 13. Full pipeline for one video
# ============================================================

def analyze_interest_zones_v5(
    video_path,
    manual_roi_polygon=None,
    render_video=True,
    max_render_frames=None,
    save_debug=True,
    # candidate filters
    raw_score_thr=0.05,
    raw_min_action_signal=0.05,
    raw_min_extension=0.18,
    # event filters
    event_min_extension=0.32,
    event_min_action_signal=0.12,
    min_event_score=0.015,
    max_gap_frames=18,
    spatial_jump_px=130,
    peak_top_k=1,
    projection_alpha=0.45,
    # heatmap parameters
    heatmap_radius_px=None,
    heatmap_blur_px=0,
    zone_relative_thr=0.24,
    zone_percentile_thr=76,
    zone_merge_dilate_px=24,
    min_component_area=350,
    pose_chunk_path=None,
    rgb_chunk_path=None,
    rtmw_frame_path=None,
    normalizer_path=None,
    pred_df=None,
    progress_callback=None,
):
    def report_progress(fraction, stage, message):
        if progress_callback is not None:
            progress_callback(float(fraction), stage, message)

    video_path = Path(video_path)
    video_id = get_video_id(video_path)
    meta = get_video_meta(video_path)

    video_out_dir = safe_mkdir(OUT_DIR / video_id)

    print("=" * 80)
    print("video_id:", video_id)
    print("meta:", meta)

    if pred_df is None:
        report_progress(0.0, "Классификация действий", "Запуск C6 LateFusion Gated BiGRU")
        pred_df = predict_actions_c6(
            video_path,
            pose_chunk_path=pose_chunk_path,
            rgb_chunk_path=rgb_chunk_path,
            normalizer_path=normalizer_path,
        )
    else:
        pred_df = pred_df.copy()
        report_progress(0.0, "Локализация кистей", "Используются рассчитанные вероятности действий")
    pred_path = video_out_dir / f"{video_id}_c6_predictions.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    report_progress(0.25, "Классификация действий", "Вероятности действий рассчитаны")
    candidates_df = build_raw_hand_candidates(
        video_path=video_path,
        pred_df=pred_df,
        raw_score_thr=raw_score_thr,
        min_action_signal=raw_min_action_signal,
        min_extension=raw_min_extension,
        rtmw_frame_path=rtmw_frame_path,
    )
    candidates_path = video_out_dir / f"{video_id}_raw_hand_candidates.csv"
    candidates_df.to_csv(candidates_path, index=False, encoding="utf-8-sig")
    summarize_candidates(candidates_df)

    report_progress(0.45, "Локализация кистей", "Кандидаты событий кистей сформированы")
    roi_polygon, roi_source = get_shelf_roi(
        meta=meta,
        candidates_df=candidates_df,
        manual_polygon=manual_roi_polygon,
    )
    print("ROI source:", roi_source)
    print("ROI polygon:", roi_polygon.tolist())

    with open(video_out_dir / f"{video_id}_shelf_roi.json", "w", encoding="utf-8") as f:
        json.dump({"roi_source": roi_source, "polygon": roi_polygon.tolist()}, f, ensure_ascii=False, indent=2)

    report_progress(0.55, "Подготовка ROI", "Область полки подготовлена")
    event_points_df = build_event_points_from_candidates(
        candidates_df=candidates_df,
        meta=meta,
        roi_polygon=roi_polygon,
        min_extension=event_min_extension,
        min_action_signal=event_min_action_signal,
        min_event_score=min_event_score,
        max_gap_frames=max_gap_frames,
        spatial_jump_px=spatial_jump_px,
        peak_top_k=peak_top_k,
        projection_alpha=projection_alpha,
    )
    event_path = video_out_dir / f"{video_id}_event_points.csv"
    event_points_df.to_csv(event_path, index=False, encoding="utf-8-sig")

    print("Event points:", len(event_points_df))
    if len(event_points_df) > 0:
        display(event_points_df.head(20))

    report_progress(0.70, "Фильтрация событий", "События рук отобраны и спроецированы в ROI")
    heatmap, roi_mask = build_interest_heatmap(
        event_points_df=event_points_df,
        meta=meta,
        roi_polygon=roi_polygon,
        radius_px=heatmap_radius_px,
        blur_px=heatmap_blur_px,
    )

    heatmap_npy_path = video_out_dir / f"{video_id}_heatmap.npy"
    np.save(heatmap_npy_path, heatmap)

    heatmap_png_path = video_out_dir / f"{video_id}_heatmap_uint8.png"
    cv2.imwrite(str(heatmap_png_path), normalize_heatmap_0_255(heatmap))

    report_progress(0.80, "Тепловая карта", "Heatmap построена, выполняется выделение зон")
    zones_df, binary_mask, used_thr, labels, zone_masks = extract_heatmap_zones(
        heatmap=heatmap,
        roi_mask=roi_mask,
        event_points_df=event_points_df,
        min_component_area=min_component_area,
        relative_thr=zone_relative_thr,
        percentile_thr=zone_percentile_thr,
        merge_dilate_px=zone_merge_dilate_px,
    )

    zones_path = video_out_dir / f"{video_id}_heatmap_zones.csv"
    zones_df.to_csv(zones_path, index=False, encoding="utf-8-sig")

    binary_path = video_out_dir / f"{video_id}_zone_binary_mask.png"
    cv2.imwrite(str(binary_path), binary_mask)

    print("Heatmap max:", float(heatmap.max()))
    print("Heatmap threshold:", used_thr)
    print("Zones:", len(zones_df))
    if len(zones_df) > 0:
        display(zones_df)

    report_progress(0.90, "Зоны интереса", "Зоны выделены, создается итоговая визуализация")
    summary_path = video_out_dir / f"{video_id}_summary_heatmap_zones.png"
    save_summary_image(
        video_path=video_path,
        heatmap=heatmap,
        roi_polygon=roi_polygon,
        zones_df=zones_df,
        event_points_df=event_points_df,
        out_path=summary_path,
        frame_idx=0,
    )
    safe_display_image(summary_path)

    video_result_path = None
    if render_video:
        report_progress(0.95, "Рендеринг видео", "Создание видео с тепловой картой")
        raw_video_path = video_out_dir / f"{video_id}_interest_v5_raw.mp4"
        video_result_path = render_interest_video(
            video_path=video_path,
            heatmap=heatmap,
            roi_polygon=roi_polygon,
            zones_df=zones_df,
            event_points_df=event_points_df,
            out_path=raw_video_path,
            max_frames=max_render_frames,
        )
        print("Rendered video:", video_result_path)
        safe_display_video(video_result_path, width=900)

    report_progress(1.0, "Зоны интереса", "Анализ видео завершен")
    result = {
        "video_id": video_id,
        "meta": meta,
        "pred_df": pred_df,
        "candidates_df": candidates_df,
        "roi_polygon": roi_polygon,
        "roi_source": roi_source,
        "event_points_df": event_points_df,
        "heatmap": heatmap,
        "roi_mask": roi_mask,
        "zones_df": zones_df,
        "paths": {
            "predictions": pred_path,
            "raw_candidates": candidates_path,
            "event_points": event_path,
            "heatmap_npy": heatmap_npy_path,
            "heatmap_png": heatmap_png_path,
            "zones": zones_path,
            "summary_image": summary_path,
            "video": video_result_path,
        },
    }
    return result


# ============================================================
# Web patch: safer shelf projection and reliable action weights
# ============================================================
ZONE_ACTION_WEIGHTS = {
    "background": 0.0,
    "reach_to_shelf": 1.30,
    "retract_from_shelf": 0.55,
    "hand_in_shelf": 1.55,
    "inspect_product": 0.0,
    "inspect_shelf": 0.20,
}

SHELF_DIRECTED_ACTIONS = [
    "reach_to_shelf",
    "hand_in_shelf",
    "retract_from_shelf",
    "inspect_shelf",
]


def _point_in_polygon_cv(point_xy, polygon):
    pts = np.asarray(polygon, dtype=np.float32)
    x, y = float(point_xy[0]), float(point_xy[1])
    return cv2.pointPolygonTest(pts, (x, y), False) >= 0


def _nearest_point_on_segment_safe(p, a, b):
    p = np.asarray(p, dtype=np.float32)
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-6:
        return a, float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / denom)
    t = float(np.clip(t, 0.0, 1.0))
    q = a + t * ab
    return q, float(np.linalg.norm(p - q))


def _nearest_point_on_polygon_safe(point_xy, polygon):
    pts = np.asarray(polygon, dtype=np.float32)
    best_q = None
    best_d = 1e18
    for i in range(len(pts)):
        q, d = _nearest_point_on_segment_safe(point_xy, pts[i], pts[(i + 1) % len(pts)])
        if d < best_d:
            best_q = q
            best_d = d
    return best_q, best_d


def project_hand_to_shelf(
    hand_xy,
    torso_xy,
    roi_polygon,
    projection_alpha=0.16,
    max_projection_alpha=0.30,
    snap_distance=35,
    max_projection_px=48,
    max_dx_px=28,
    max_up_px=55,
    max_down_px=8,
    **kwargs,
):
    """
    Safer projection for web/report mode.
    Keeps the interest point close to the detected hand and avoids jumps into neighbouring shelf areas.
    """
    hand_xy = np.asarray(hand_xy, dtype=np.float32)
    torso_xy = np.asarray(torso_xy, dtype=np.float32)

    if not np.all(np.isfinite(hand_xy)) or not np.all(np.isfinite(torso_xy)):
        return None, "bad_xy"

    vec = hand_xy - torso_xy
    dist = float(np.linalg.norm(vec))
    if dist < 1e-6:
        return None, "bad_direction"

    unit = vec / dist
    hand_inside = _point_in_polygon_cv(hand_xy, roi_polygon)

    alpha = float(min(projection_alpha, max_projection_alpha))
    step_px = min(max_projection_px, alpha * dist)
    if hand_inside:
        step_px = min(step_px, 34)

    candidate = hand_xy + unit * step_px
    dx = float(np.clip(candidate[0] - hand_xy[0], -max_dx_px, max_dx_px))
    dy = float(candidate[1] - hand_xy[1])
    dy = float(np.clip(dy, -max_up_px, max_down_px))

    candidate = np.array([hand_xy[0] + dx, hand_xy[1] + dy], dtype=np.float32)

    if _point_in_polygon_cv(candidate, roi_polygon):
        return candidate, f"safe_project_{alpha:.2f}_{step_px:.0f}px"

    if hand_inside:
        return hand_xy.astype(np.float32), "inside_use_raw_hand"

    nearest, nearest_dist = _nearest_point_on_polygon_safe(hand_xy, roi_polygon)
    if nearest_dist <= snap_distance:
        return nearest.astype(np.float32), f"near_roi_snap_{nearest_dist:.1f}px"

    return None, "outside_roi"
