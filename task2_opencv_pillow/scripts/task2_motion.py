"""task-2 题 4：example2.mp4 运动目标检测 —— MOG2 背景减除 vs 帧差法（无背景模型）.

对比方案：
* 方法 A：MOG2 高斯混合背景建模 + 背景减除（OpenCV BackgroundSubtractorMOG2）；
* 方法 B：相邻帧差分（frame differencing），不建立背景模型，属于“未使用背景减除”的基线方法。

输出：
    outputs/videos/example2_mog2.mp4
    outputs/videos/example2_framediff.mp4
    outputs/videos/example2_comparison.mp4
    outputs/images/fig_task2_4_motion_compare.png
    outputs/images/fig_task2_4_motion_timeseries.png
    outputs/stats/task2_motion_stats.json

运行：
    python task2_opencv_pillow/scripts/task2_motion.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "task2"
OUT = ROOT / "task2_opencv_pillow" / "outputs"
VID_OUT = OUT / "videos"
IMG_OUT = OUT / "images"
STAT_OUT = OUT / "stats"

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

MIN_AREA = 300


def clean_mask(mask: np.ndarray) -> np.ndarray:
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k7)
    return mask


def detect_boxes(mask: np.ndarray, min_area: int = MIN_AREA):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes, fills = [], []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < 8 or h < 8:
            continue
        boxes.append((x, y, w, h, area))
        fills.append(area / max(w * h, 1))
    return boxes, fills


def draw_boxes(frame: np.ndarray, boxes, color, tag: str) -> np.ndarray:
    vis = frame.copy()
    for (x, y, w, h, area) in boxes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
    cv2.rectangle(vis, (10, 10), (330, 52), (0, 0, 0), -1)
    cv2.putText(vis, f"{tag}  objects={len(boxes)}", (18, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return vis


def make_panel(frame: np.ndarray, mask: np.ndarray, boxes, tag: str,
               color=(0, 255, 0)) -> np.ndarray:
    """把掩膜以半透明颜色叠加到画面上，并画出检测框。"""
    vis = frame.copy()
    color_layer = np.zeros_like(vis)
    color_layer[mask > 0] = color
    vis = cv2.addWeighted(vis, 1.0, color_layer, 0.45, 0)
    vis = draw_boxes(vis, boxes, color, tag)
    return cv2.resize(vis, (640, 360), interpolation=cv2.INTER_AREA)


def main() -> None:
    for d in (VID_OUT, IMG_OUT, STAT_OUT):
        d.mkdir(parents=True, exist_ok=True)

    src = DATA / "example2.mp4"
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise FileNotFoundError(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw_mog = cv2.VideoWriter(str(VID_OUT / "example2_mog2.mp4"), fourcc, fps, (w, h))
    vw_diff = cv2.VideoWriter(str(VID_OUT / "example2_framediff.mp4"), fourcc, fps, (w, h))
    vw_cmp = cv2.VideoWriter(str(VID_OUT / "example2_comparison.mp4"), fourcc, fps,
                             (640 * 3, 360))

    mog = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25,
                                             detectShadows=True)
    prev_gray = None
    key_frames = {}

    records = {"mog2": [], "framediff": []}
    masks_prev = {"mog2": None, "framediff": None}
    mask_iou = {"mog2": [], "framediff": []}
    t0 = time.perf_counter()

    for idx in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # ---------- A. MOG2 ----------
        fg = mog.apply(frame)
        fg[fg == 127] = 0          # 去掉阴影（MOG2 阴影标记为 127）
        fg_mog = clean_mask(fg)
        boxes_mog, fills_mog = detect_boxes(fg_mog)

        # ---------- B. 帧差法（无背景减除） ----------
        if prev_gray is None:
            fg_diff = np.zeros_like(gray)
        else:
            diff = cv2.absdiff(gray_blur, prev_gray)
            _, fg_diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            fg_diff = clean_mask(fg_diff)
        prev_gray = gray_blur
        boxes_diff, fills_diff = detect_boxes(fg_diff)

        # 掩膜的时间稳定性（相邻帧前景 IoU）
        for name, msk in (("mog2", fg_mog), ("framediff", fg_diff)):
            prev_m = masks_prev[name]
            if prev_m is not None:
                inter = np.count_nonzero((msk > 0) & (prev_m > 0))
                union = np.count_nonzero((msk > 0) | (prev_m > 0))
                mask_iou[name].append(inter / union if union else 1.0)
            masks_prev[name] = msk.copy()

        records["mog2"].append({
            "frame": idx,
            "fg_ratio": float(np.count_nonzero(fg_mog) / fg_mog.size),
            "n_objects": len(boxes_mog),
            "mean_fill": float(np.mean(fills_mog)) if fills_mog else 0.0,
            "mean_box_area": float(np.mean([b[4] for b in boxes_mog])) if boxes_mog else 0.0,
        })
        records["framediff"].append({
            "frame": idx,
            "fg_ratio": float(np.count_nonzero(fg_diff) / fg_diff.size),
            "n_objects": len(boxes_diff),
            "mean_fill": float(np.mean(fills_diff)) if fills_diff else 0.0,
            "mean_box_area": float(np.mean([b[4] for b in boxes_diff])) if boxes_diff else 0.0,
        })

        vis_mog = draw_boxes(frame, boxes_mog, (0, 255, 0), "MOG2")
        vis_diff = draw_boxes(frame, boxes_diff, (0, 165, 255), "Frame diff")
        vw_mog.write(vis_mog)
        vw_diff.write(vis_diff)
        cmp_frame = np.hstack([
            cv2.resize(draw_boxes(frame, [], (255, 255, 255), "Original"),
                       (640, 360), interpolation=cv2.INTER_AREA),
            make_panel(frame, fg_mog, boxes_mog, "MOG2", (0, 255, 0)),
            make_panel(frame, fg_diff, boxes_diff, "Frame diff (no BG model)",
                       (0, 165, 255)),
        ])
        vw_cmp.write(cmp_frame)

        if idx in (150, 400, 700, 930):
            key_frames[idx] = (frame.copy(), fg_mog.copy(), boxes_mog,
                               fg_diff.copy(), boxes_diff)

        if idx % 100 == 0:
            print(f"  frame {idx}/{n_frames}  mog2={len(boxes_mog)}  "
                  f"diff={len(boxes_diff)}")

    elapsed = time.perf_counter() - t0
    cap.release()
    for vw in (vw_mog, vw_diff, vw_cmp):
        vw.release()

    # ---------- 统计汇总 ----------
    def summarize(name: str):
        recs = records[name][30:]  # 跳过背景模型预热阶段
        ious = mask_iou[name][30:]
        return {
            "avg_objects": float(np.mean([r["n_objects"] for r in recs])),
            "max_objects": int(np.max([r["n_objects"] for r in recs])),
            "avg_fg_ratio_percent": float(np.mean([r["fg_ratio"] for r in recs]) * 100),
            "avg_bbox_fill": float(np.mean([r["mean_fill"] for r in recs if r["n_objects"]])),
            "avg_box_area_px": float(np.mean([r["mean_box_area"] for r in recs
                                              if r["n_objects"]])),
            "avg_mask_iou_consecutive": float(np.mean(ious)) if ious else 0.0,
        }

    summary = {
        "video": src.name,
        "fps": round(fps, 2),
        "frames": n_frames,
        "size": [w, h],
        "processing_seconds": round(elapsed, 2),
        "processing_fps": round(n_frames / elapsed, 2),
        "mog2": summarize("mog2"),
        "framediff": summarize("framediff"),
    }
    with open(STAT_OUT / "task2_motion_stats.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---------- 关键帧对比图 ----------
    fig, axes = plt.subplots(len(key_frames), 3, figsize=(16, 3.6 * len(key_frames)))
    for row, (idx, (frame, m1, b1, m2, b2)) in enumerate(key_frames.items()):
        axes[row, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        axes[row, 0].set_title(f"原视频 第 {idx} 帧", fontsize=11)
        axes[row, 1].imshow(m1, cmap="gray")
        axes[row, 1].set_title(f"MOG2 前景掩膜（{len(b1)} 个目标）", fontsize=11)
        axes[row, 2].imshow(m2, cmap="gray")
        axes[row, 2].set_title(f"帧差法前景掩膜（{len(b2)} 个目标）", fontsize=11)
        for ax in axes[row]:
            ax.axis("off")
    fig.suptitle("MOG2 背景减除 与 帧差法（无背景模型）对比", fontsize=15)
    fig.tight_layout()
    fig.savefig(IMG_OUT / "fig_task2_4_motion_compare.png", dpi=140,
                bbox_inches="tight")
    plt.close(fig)

    # ---------- 逐帧曲线 ----------
    frames = np.array([r["frame"] for r in records["mog2"]])
    n_mog = np.array([r["n_objects"] for r in records["mog2"]])
    n_diff = np.array([r["n_objects"] for r in records["framediff"]])
    fg_mog = np.array([r["fg_ratio"] for r in records["mog2"]]) * 100
    fg_diff = np.array([r["fg_ratio"] for r in records["framediff"]]) * 100

    def smooth(x, k=15):
        return np.convolve(x, np.ones(k) / k, mode="same")

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.2))
    axes[0].plot(frames, n_mog, color="#2ca02c", alpha=0.35, label="MOG2 原始")
    axes[0].plot(frames, smooth(n_mog), color="#2ca02c", lw=2, label="MOG2 滑动平均")
    axes[0].plot(frames, n_diff, color="#ff7f0e", alpha=0.35, label="帧差法 原始")
    axes[0].plot(frames, smooth(n_diff), color="#ff7f0e", lw=2, label="帧差法 滑动平均")
    axes[0].set_xlabel("帧序号")
    axes[0].set_ylabel("检测目标数")
    axes[0].set_title("每帧检测到的运动目标数量")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(frames, fg_mog, color="#2ca02c", alpha=0.4, label="MOG2")
    axes[1].plot(frames, fg_diff, color="#ff7f0e", alpha=0.4, label="帧差法")
    axes[1].set_xlabel("帧序号")
    axes[1].set_ylabel("前景像素占比 (%)")
    axes[1].set_title("前景区域占全图比例")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMG_OUT / "fig_task2_4_motion_timeseries.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[完成] 对比视频: {VID_OUT / 'example2_comparison.mp4'}")


if __name__ == "__main__":
    main()
