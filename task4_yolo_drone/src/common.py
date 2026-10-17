"""task-4 题目一（YOLO 无人机目标检测）公共路径与工具。"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "task4"
DATASET_DIR = ROOT / "data" / "drone_dataset"
OUT = ROOT / "task4_yolo_drone" / "outputs"
MODEL_OUT = OUT / "models"
PRED_OUT = OUT / "predictions"
VID_OUT = OUT / "videos"
TRAIN_OUT = OUT / "train"
FIG_OUT = OUT / "figures"
STAT_OUT = OUT / "stats"

for _d in (OUT, MODEL_OUT, PRED_OUT, VID_OUT, TRAIN_OUT, FIG_OUT, STAT_OUT):
    _d.mkdir(parents=True, exist_ok=True)


def find_data_yaml() -> Path:
    """自动定位 Roboflow 数据集中的 data.yaml。"""
    direct = DATASET_DIR / "data.yaml"
    if direct.exists():
        return direct
    candidates = sorted(DATASET_DIR.rglob("data.yaml"))
    if not candidates:
        raise FileNotFoundError(
            f"未找到 data.yaml，请先将无人机数据集解压到 {DATASET_DIR}")
    return candidates[0]


def load_yaml(path: Path) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_best_weights() -> Path:
    """优先使用训练元数据中记录的 best.pt，其次搜索最新训练结果。"""
    meta = STAT_OUT / "train_result.json"
    if meta.exists():
        p = Path(load_json(meta).get("best_weights", ""))
        if p.exists():
            return p
    candidates = sorted(TRAIN_OUT.rglob("best.pt"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("未找到训练权重 best.pt，请先运行 02_train.py")
    return candidates[0]


def imread_unicode(path: Path) -> np.ndarray:
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def imwrite_unicode(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, img)
    if not ok:
        raise RuntimeError(path)
    buf.tofile(str(path))


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def file_size_mb(path: Path) -> float:
    return round(path.stat().st_size / (1024 * 1024), 2)
