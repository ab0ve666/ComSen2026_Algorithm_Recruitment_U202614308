"""task-4 题目一（2）：使用 YOLO 官方预训练权重对 example 下三张图片推理。

官方权重：yolo11n.pt（COCO 80 类），下载后放在 data/task4/weights/。
输出：标注图片、检测结果 JSON、可视化汇总图。

运行：
    python task4_yolo_drone/src/01_pretrained_inference.py
"""
from __future__ import annotations

import argparse
from collections import Counter

import cv2
from ultralytics import YOLO

from common import (DATA, FIG_OUT, PRED_OUT, STAT_OUT, find_data_yaml,
                    imread_unicode, imwrite_unicode, save_json,
                    setup_matplotlib)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str,
                        default=str(DATA / "weights" / "yolo11n.pt"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    weights = DATA / "weights" / "yolo11n.pt"
    model = YOLO(str(weights))
    print(f"[模型] {weights}  类别数={len(model.names)}")

    images = sorted((DATA / "example").glob("*.jpg"))
    if not images:
        raise FileNotFoundError("example 目录下没有图片")
    results_all = {}
    annotated = []
    class_counter = Counter()

    for img_path in images:
        img = imread_unicode(img_path)
        result = model.predict(img, conf=args.conf, imgsz=args.imgsz,
                               verbose=False)[0]
        detections = []
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            name = model.names[cls_id]
            detections.append({
                "class_id": cls_id,
                "class_name": name,
                "confidence": round(conf, 4),
                "xyxy": [round(v, 1) for v in xyxy],
                "wh": [round(xyxy[2] - xyxy[0], 1),
                       round(xyxy[3] - xyxy[1], 1)],
            })
            class_counter[name] += 1
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        results_all[img_path.name] = {
            "image_size": [img.shape[1], img.shape[0]],
            "num_detections": len(detections),
            "detections": detections,
        }
        vis = result.plot(line_width=2, font_size=12)
        out_path = PRED_OUT / f"pretrained_{img_path.stem}.jpg"
        imwrite_unicode(out_path, vis)
        annotated.append((img_path.name, cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                          len(detections)))
        print(f"[推理] {img_path.name}: {len(detections)} 个目标  -> {out_path.name}")

    save_json(STAT_OUT / "pretrained_inference.json", {
        "weights": weights.name,
        "conf_threshold": args.conf,
        "imgsz": args.imgsz,
        "num_classes": len(model.names),
        "class_counts": dict(class_counter),
        "images": results_all,
    })

    # 汇总图
    plt = setup_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for ax, (name, rgb, n) in zip(axes.ravel()[:3], annotated):
        ax.imshow(rgb)
        ax.set_title(f"{name}：{n} 个检测目标", fontsize=11)
        ax.axis("off")
    ax = axes.ravel()[3]
    names = list(class_counter.keys())
    values = [class_counter[k] for k in names]
    if names:
        ax.barh(names, values, color="#4c72b0")
        ax.set_xlabel("检测框数量")
        ax.set_title("COCO 预训练模型在三张示例图上的类别统计", fontsize=11)
        for i, v in enumerate(values):
            ax.text(v + 0.03, i, str(v), va="center", fontsize=9)
        ax.grid(axis="x", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "未检测到目标", ha="center")
        ax.axis("off")
    fig.suptitle("task-4 题目一：YOLO11n 官方预训练权重推理结果", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIG_OUT / "fig_task4_1_pretrained.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    print(f"[完成] 检测 JSON: {STAT_OUT / 'pretrained_inference.json'}")
    print(f"[完成] 汇总图: {FIG_OUT / 'fig_task4_1_pretrained.png'}")


if __name__ == "__main__":
    main()
