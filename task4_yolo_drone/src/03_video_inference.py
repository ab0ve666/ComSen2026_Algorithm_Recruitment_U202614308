"""task-4 题目一（4）：对无人机飞行视频 example.mp4 做推理并分析指标。

同时比较：
* 题目提供的初始权重 drone.pt；
* 在无人机数据集上微调后的 best.pt。

统计：模型大小、纯推理 FPS、端到端视频处理 FPS、检测覆盖率、平均置信度、
目标框面积占比、相邻帧重叠稳定性（EAO 的替代指标，EAO 需要真值标注）。

运行：
    python task4_yolo_drone/src/03_video_inference.py
"""
from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from common import (DATA, FIG_OUT, OUT, STAT_OUT, VID_OUT, file_size_mb,
                    find_best_weights, save_json, setup_matplotlib)


def iou_xyxy(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def analyze_video(weights: Path, tag: str, src: Path, conf: float,
                  imgsz: int, device: str) -> dict:
    model = YOLO(str(weights))
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    writer = cv2.VideoWriter(str(VID_OUT / f"{tag}.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    records = []
    trail = []
    sample_frames = {}
    infer_ms, pre_ms, post_ms = [], [], []
    t_start = time.perf_counter()
    idx = 0

    stream = model.predict(source=str(src), stream=True, conf=conf, iou=0.5,
                           imgsz=imgsz, device=device, verbose=False)
    for result in stream:
        frame = result.orig_img.copy()
        boxes = result.boxes
        dets = []
        for b in boxes:
            xyxy = b.xyxy[0].tolist()
            dets.append({
                "class": model.names[int(b.cls[0])],
                "conf": float(b.conf[0]),
                "xyxy": xyxy,
            })
        dets.sort(key=lambda d: d["conf"], reverse=True)
        best = dets[0] if dets else None
        if best:
            x1, y1, x2, y2 = [int(v) for v in best["xyxy"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label = f"{best['class']} {best['conf']:.2f}"
            cv2.putText(frame, label, (x1, max(30, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            trail.append((cx, cy))
        if len(trail) > 1:
            cv2.polylines(frame, [np.array(trail, np.int32).reshape(-1, 1, 2)],
                          False, (0, 200, 255), 2)
        cv2.putText(frame, f"{tag}  frame {idx}  objects={len(dets)}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        writer.write(frame)

        if idx in (0, 90, 180, 270):
            sample_frames[idx] = frame.copy()
        area_ratio = 0.0
        if best:
            x1, y1, x2, y2 = best["xyxy"]
            area_ratio = (x2 - x1) * (y2 - y1) / (w * h)
        records.append({
            "frame": idx,
            "num": len(dets),
            "conf": best["conf"] if best else 0.0,
            "area_ratio": area_ratio,
            "xyxy": best["xyxy"] if best else None,
            "class": best["class"] if best else None,
        })
        speed = result.speed or {}
        if "inference" in speed:
            infer_ms.append(speed["inference"])
            pre_ms.append(speed.get("preprocess", 0.0))
            post_ms.append(speed.get("postprocess", 0.0))
        idx += 1
    writer.release()
    elapsed = time.perf_counter() - t_start

    # 检测连续性 / 相邻帧重叠
    seg_lengths, cur = [], 0
    overlaps = []
    prev_box = None
    for r in records:
        if r["num"] > 0:
            cur += 1
            if prev_box is not None and r["xyxy"] is not None:
                overlaps.append(iou_xyxy(prev_box, r["xyxy"]))
            prev_box = r["xyxy"]
        else:
            if cur:
                seg_lengths.append(cur)
            cur = 0
            prev_box = None
    if cur:
        seg_lengths.append(cur)
    confs = [r["conf"] for r in records if r["num"] > 0]
    class_counts = Counter(r["class"] for r in records if r["class"])

    stats = {
        "weights": str(weights),
        "model_size_mb": file_size_mb(weights),
        "video": src.name,
        "fps_video": round(fps, 2),
        "frames": len(records),
        "resolution": [w, h],
        "conf_threshold": conf,
        "detection_coverage_percent": round(
            sum(1 for r in records if r["num"] > 0) / max(len(records), 1) * 100, 2),
        "avg_objects_per_frame": round(float(np.mean([r["num"] for r in records])), 3),
        "avg_confidence": round(float(np.mean(confs)), 4) if confs else 0.0,
        "avg_area_ratio_percent": round(
            float(np.mean([r["area_ratio"] for r in records if r["num"] > 0])) * 100, 3)
        if confs else 0.0,
        "class_counts": dict(class_counts),
        "mean_consecutive_box_iou": round(float(np.mean(overlaps)), 4)
        if overlaps else 0.0,
        "max_continuous_detection_frames": max(seg_lengths) if seg_lengths else 0,
        "num_detection_segments": len(seg_lengths),
        "pure_inference_ms": round(float(np.mean(infer_ms)), 3) if infer_ms else None,
        "pure_inference_fps": round(1000.0 / float(np.mean(infer_ms)), 2)
        if infer_ms else None,
        "preprocess_ms": round(float(np.mean(pre_ms)), 3) if pre_ms else None,
        "postprocess_ms": round(float(np.mean(post_ms)), 3) if post_ms else None,
        "end_to_end_fps": round(len(records) / elapsed, 2),
        "elapsed_seconds": round(elapsed, 2),
        "ea_o_note": "EAO 需要逐帧真值标注，本实验无真值；以相邻帧框 IoU 与连续检测段长度作为跟踪稳定性替代指标",
        "records": records,
    }
    save_json(STAT_OUT / f"video_inference_{tag}.json", stats)
    print(f"[{tag}] coverage={stats['detection_coverage_percent']}% "
          f"conf={stats['avg_confidence']} infer={stats['pure_inference_fps']} FPS "
          f"e2e={stats['end_to_end_fps']} FPS")
    return stats, sample_frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default=str(DATA / "drone.pt"))
    parser.add_argument("--best", type=str, default="")
    parser.add_argument("--source", type=str,
                        default=str(DATA / "example" / "example.mp4"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    best = Path(args.best) if args.best else find_best_weights()
    source = Path(args.source)
    print(f"[base] {args.base}")
    print(f"[best] {best}")

    base_stats, _ = analyze_video(Path(args.base), "example_drone_base", source,
                                  args.conf, args.imgsz, args.device)
    best_stats, samples = analyze_video(best, "example_drone_best", source,
                                        args.conf, args.imgsz, args.device)

    plt = setup_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    for ax, (idx, frame) in zip(axes.ravel(), sorted(samples.items())):
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.set_title(f"微调模型推理：第 {idx} 帧", fontsize=10)
        ax.axis("off")
    fig.suptitle("YOLO 无人机视频推理结果（best.pt）", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIG_OUT / "fig_task4_3_video_samples.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.2))
    for stats, color, label in ((base_stats, "#1f77b4", "drone.pt"),
                                (best_stats, "#ff7f0e", "best.pt（微调）")):
        frames = [r["frame"] for r in stats["records"]]
        confs = [r["conf"] if r["num"] else np.nan for r in stats["records"]]
        axes[0].plot(frames, confs, lw=1.0, color=color, label=label)
        axes[1].plot(frames, [r["area_ratio"] * 100 if r["num"] else np.nan
                              for r in stats["records"]],
                     lw=1.0, color=color, label=label)
    axes[0].set_xlabel("帧序号")
    axes[0].set_ylabel("最高置信度")
    axes[0].set_title("逐帧检测置信度")
    axes[0].set_ylim(0, 1.02)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("帧序号")
    axes[1].set_ylabel("目标框面积占比 (%)")
    axes[1].set_title("逐帧目标框面积占比")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_OUT / "fig_task4_3_video_timeline.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    summary = {
        "base": {k: v for k, v in base_stats.items() if k != "records"},
        "best": {k: v for k, v in best_stats.items() if k != "records"},
        "comparison": {
            "coverage_gain_percent": round(
                best_stats["detection_coverage_percent"]
                - base_stats["detection_coverage_percent"], 2),
            "confidence_gain": round(
                best_stats["avg_confidence"] - base_stats["avg_confidence"], 4),
            "fps_gain": round(
                (best_stats["end_to_end_fps"] or 0)
                - (base_stats["end_to_end_fps"] or 0), 2),
        },
    }
    save_json(STAT_OUT / "video_inference_summary.json", summary)
    print(f"[完成] 视频: {VID_OUT / 'example_drone_best.mp4'}")


if __name__ == "__main__":
    main()
