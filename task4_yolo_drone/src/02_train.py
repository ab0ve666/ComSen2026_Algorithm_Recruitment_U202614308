"""task-4 题目一（3）：基于无人机数据集与 drone.pt 预训练权重训练 YOLO11。

流程：
1. 读取 Roboflow YOLO 格式数据集（data/drone_dataset），统计与可视化；
2. 以 data/task4/drone.pt 为初始权重微调（默认 40 epochs，640×640，AMP）；
3. 保存 best.pt / last.pt，输出 Ultralytics 训练曲线、混淆矩阵等；
4. 在 val / test 上评估 mAP，并计算“匹配框分类准确率”。

运行：
    python task4_yolo_drone/src/02_train.py --epochs 40 --batch 16
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from common import (DATASET_DIR, DATA, FIG_OUT, OUT, STAT_OUT, TRAIN_OUT,
                    find_data_yaml, imwrite_unicode, load_yaml,
                    save_json, setup_matplotlib)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: Path):
    """只返回图片文件，忽略 Ultralytics 生成的 .npy 缓存。"""
    if not folder.exists():
        return []
    return sorted(p for p in folder.glob("*") if p.suffix.lower() in IMG_EXTS)


def count_split(root: Path, split: str):
    img_dir = root / split / "images"
    lbl_dir = root / split / "labels"
    images = list_images(img_dir)
    labels = sorted(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []
    counter = Counter()
    for lbl in labels:
        for line in lbl.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if parts:
                counter[int(parts[0])] += 1
    return images, labels, counter


def yolo_to_xyxy(parts, w, h):
    cls = int(parts[0])
    cx, cy, bw, bh = map(float, parts[1:5])
    return (cls, (cx - bw / 2) * w, (cy - bh / 2) * h,
            (cx + bw / 2) * w, (cy + bh / 2) * h)


def iou_xyxy(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def visualize_dataset(root: Path, names: dict, split: str = "train"):
    plt = setup_matplotlib()
    rng = random.Random(2026)
    img_dir = root / split / "images"
    lbl_dir = root / split / "labels"
    images = list_images(img_dir)
    if not images:
        return
    samples = rng.sample(images, min(6, len(images)))
    colors = [(255, 56, 56), (56, 200, 255), (60, 220, 120), (200, 80, 255)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, img_path in zip(axes.ravel(), samples):
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8),
                           cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        lbl = lbl_dir / f"{img_path.stem}.txt"
        if lbl.exists():
            for line in lbl.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls, x1, y1, x2, y2 = yolo_to_xyxy(parts, w, h)
                color = colors[cls % len(colors)]
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                              color, 2)
                cv2.putText(img, names.get(cls, str(cls)),
                            (int(x1), max(18, int(y1) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{img_path.name}  ({w}×{h})", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"无人机数据集 {split} 划分样例与标注框", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_OUT / f"fig_task4_2_dataset_{split}.png", dpi=140,
                bbox_inches="tight")
    plt.close(fig)


def matched_accuracy(model: YOLO, root: Path, split: str, conf=0.25,
                     iou_thr=0.5) -> dict:
    """计算匹配框的类别准确率、召回率（检测任务里“准确率”的合理替代）。"""
    split_dir = split
    if not (root / split / "images").exists() and split == "val" \
            and (root / "valid" / "images").exists():
        split_dir = "valid"
    img_dir = root / split_dir / "images"
    lbl_dir = root / split_dir / "labels"
    images = list_images(img_dir)
    total_gt, total_pred, matched, correct = 0, 0, 0, 0
    # 行：真实类别（bird, drone, background），列：预测类别（同）
    cm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for img_path in images:
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8),
                           cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        gt = []
        lbl = lbl_dir / f"{img_path.stem}.txt"
        if lbl.exists():
            for line in lbl.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    cls, x1, y1, x2, y2 = yolo_to_xyxy(parts, w, h)
                    gt.append((cls, x1, y1, x2, y2))
        res = model.predict(img, conf=conf, imgsz=640, verbose=False)[0]
        preds = [(int(b.cls[0]), *b.xyxy[0].tolist()) for b in res.boxes]
        total_gt += len(gt)
        total_pred += len(preds)

        # 贪心匹配：按 IoU 从大到小配对
        pairs = []
        for gi, g in enumerate(gt):
            for pi, p in enumerate(preds):
                v = iou_xyxy(g[1:], p[1:])
                if v >= iou_thr:
                    pairs.append((v, gi, pi))
        pairs.sort(reverse=True)
        used_g, used_p = set(), set()
        for _, gi, pi in pairs:
            if gi in used_g or pi in used_p:
                continue
            used_g.add(gi)
            used_p.add(pi)
            matched += 1
            if gt[gi][0] == preds[pi][0]:
                correct += 1
            cm[gt[gi][0]][preds[pi][0]] += 1
        for gi, g in enumerate(gt):
            if gi not in used_g:
                cm[g[0]][2] += 1  # 漏检
        for pi, p in enumerate(preds):
            if pi not in used_p:
                cm[2][p[0]] += 1  # 误检
    return {
        "split": split,
        "images": len(images),
        "gt_objects": total_gt,
        "pred_objects": total_pred,
        "matched_objects": matched,
        "classification_accuracy": round(correct / matched, 4) if matched else 0.0,
        "detection_recall": round(matched / total_gt, 4) if total_gt else 0.0,
        "detection_precision": round(correct / total_pred, 4) if total_pred else 0.0,
        "confusion_matrix": cm,
        "class_names": ["bird", "drone", "background"],
    }


def plot_custom_confusion(cm, class_names, path: Path):
    """绘制 conf=0.25、IoU=0.5 匹配后的 3×3 混淆矩阵（含漏检/误检）。"""
    plt = setup_matplotlib()
    cm = np.array(cm, dtype=float)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm),
                        where=row_sum > 0)
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    labels = ["bird", "drone", "background"]
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title("验证集混淆矩阵（conf=0.25，IoU=0.5 匹配）", fontsize=12)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{int(cm[i, j])}\n({cm_norm[i, j]:.2f})",
                    ha="center", va="center", fontsize=9,
                    color="white" if cm_norm[i, j] > 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, label="按真实类别归一化")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(csv_path: Path):
    plt = setup_matplotlib()
    if not csv_path.exists():
        return
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k.strip(): float(v) for k, v in row.items() if v != ""})
    if not rows:
        return
    epochs = [r.get("epoch", i + 1) for i, r in enumerate(rows)]

    def get(key):
        return [r.get(key, np.nan) for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for key, label in [("train/box_loss", "train box_loss"),
                       ("train/cls_loss", "train cls_loss"),
                       ("train/dfl_loss", "train dfl_loss"),
                       ("val/box_loss", "val box_loss"),
                       ("val/cls_loss", "val cls_loss"),
                       ("val/dfl_loss", "val dfl_loss")]:
        if key in rows[0]:
            axes[0, 0].plot(epochs, get(key), lw=1.4, label=label)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("训练 / 验证损失曲线")
    axes[0, 0].legend(fontsize=7, ncol=2)
    axes[0, 0].grid(alpha=0.3)

    for key, label in [("metrics/precision(B)", "Precision"),
                       ("metrics/recall(B)", "Recall")]:
        axes[0, 1].plot(epochs, get(key), lw=1.5, label=label)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("值")
    axes[0, 1].set_title("验证集 Precision / Recall")
    axes[0, 1].set_ylim(0, 1.02)
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    for key, label in [("metrics/mAP50(B)", "mAP@0.5"),
                       ("metrics/mAP50-95(B)", "mAP@0.5:0.95")]:
        axes[1, 0].plot(epochs, get(key), lw=1.6, label=label)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("mAP")
    axes[1, 0].set_title("验证集 mAP 曲线")
    axes[1, 0].set_ylim(0, 1.02)
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(epochs, get("metrics/mAP50(B)"), lw=1.6,
                    label="mAP@0.5（检测准确率代理）")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("mAP@0.5")
    axes[1, 1].set_title("检测精度随训练轮次变化")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    fig.suptitle("YOLO11 无人机检测训练过程", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIG_OUT / "fig_task4_2_train_metrics.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=str(DATA / "drone.pt"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--name", type=str, default="drone_yolo11n")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--eval-only", action="store_true",
                        help="跳过训练，直接评估已有 best.pt")
    args = parser.parse_args()

    yaml_path = find_data_yaml()
    cfg = load_yaml(yaml_path)
    names = {int(k): v for k, v in dict(cfg["names"]).items()} \
        if isinstance(cfg["names"], dict) else \
        {i: v for i, v in enumerate(cfg["names"])}
    root = yaml_path.parent
    print(f"[数据] {yaml_path}")
    print(f"[数据] 类别: {names}")

    stats = {"data_yaml": str(yaml_path), "names": names, "splits": {}}
    for split in ("train", "val", "test", "valid"):
        if (root / split / "images").exists():
            imgs, lbls, counter = count_split(root, split)
            stats["splits"][split] = {
                "images": len(imgs),
                "labels": len(lbls),
                "class_counts": {names.get(k, str(k)): v
                                 for k, v in sorted(counter.items())},
            }
            print(f"[数据] {split}: {len(imgs)} 张图像, "
                  f"类别分布 {stats['splits'][split]['class_counts']}")
    if "val" not in stats["splits"] and "valid" in stats["splits"]:
        stats["splits"]["val"] = stats["splits"]["valid"]
    save_json(STAT_OUT / "dataset_stats.json", stats)
    visualize_dataset(root, names, "train")
    if (root / "valid" / "images").exists():
        visualize_dataset(root, names, "valid")

    run_dir = TRAIN_OUT / args.name
    best = run_dir / "weights" / "best.pt"
    if not args.eval_only:
        model = YOLO(args.weights)
        print(f"[训练] 初始权重 {args.weights}")
        model.train(
            data=str(yaml_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=str(TRAIN_OUT),
            name=args.name,
            exist_ok=True,
            seed=42,
            patience=15,
            workers=args.workers,
            cache="disk",
            amp=True,
            plots=True,
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.01,
            weight_decay=0.0005,
            warmup_epochs=3,
            cos_lr=True,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            translate=0.1,
            scale=0.5,
            fliplr=0.5,
            mosaic=1.0,
            close_mosaic=10,
            val=True,
            save=True,
            verbose=True,
        )
    if not best.exists():
        raise FileNotFoundError(best)
    print(f"[训练] best.pt: {best}")

    # 实际完成的 epoch 数与最佳 epoch
    actual_epochs = args.epochs
    best_epoch = None
    csv_path = run_dir / "results.csv"
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        actual_epochs = len(rows)
        if rows:
            key = "metrics/mAP50(B)"
            best_row = max(rows, key=lambda x: float(x.get(key, 0) or 0))
            best_epoch = int(float(best_row.get("epoch", 0)))

    # 训练曲线
    plot_training_curves(run_dir / "results.csv")

    # 重新加载 best 做 val / test 评估
    best_model = YOLO(str(best))
    val_metrics = best_model.val(data=str(yaml_path), split="val",
                                 imgsz=args.imgsz, batch=8, device=args.device,
                                 project=str(TRAIN_OUT),
                                 name=f"{args.name}_val", exist_ok=True,
                                 plots=True, verbose=False)
    val_result = {
        "mAP50": round(float(val_metrics.box.map50), 4),
        "mAP50_95": round(float(val_metrics.box.map), 4),
        "precision": round(float(val_metrics.box.mp), 4),
        "recall": round(float(val_metrics.box.mr), 4),
    }
    print(f"[评估] val: {val_result}")
    val_acc = matched_accuracy(best_model, root, "val")
    print(f"[评估] val 匹配框分类准确率: {val_acc}")
    plot_custom_confusion(val_acc["confusion_matrix"], val_acc["class_names"],
                          FIG_OUT / "fig_task4_2_confusion.png")

    test_result, test_acc = None, None
    if (root / "test" / "images").exists():
        test_metrics = best_model.val(data=str(yaml_path), split="test",
                                      imgsz=args.imgsz, batch=8,
                                      device=args.device,
                                      project=str(TRAIN_OUT),
                                      name=f"{args.name}_test", exist_ok=True,
                                      plots=True, verbose=False)
        test_result = {
            "mAP50": round(float(test_metrics.box.map50), 4),
            "mAP50_95": round(float(test_metrics.box.map), 4),
            "precision": round(float(test_metrics.box.mp), 4),
            "recall": round(float(test_metrics.box.mr), 4),
        }
        test_acc = matched_accuracy(best_model, root, "test")
        print(f"[评估] test: {test_result}")
        print(f"[评估] test 匹配框分类准确率: {test_acc}")

    train_result = {
        "weights": args.weights,
        "best_weights": str(best),
        "run_dir": str(run_dir),
        "epochs": actual_epochs,
        "planned_epochs": args.epochs,
        "best_epoch": best_epoch,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "names": names,
        "dataset": stats,
        "val": val_result,
        "test": test_result,
        "val_matched_accuracy": val_acc,
        "test_matched_accuracy": test_acc,
        "key_plots": {
            "results_png": str(run_dir / "results.png"),
            "confusion_matrix": str(TRAIN_OUT / f"{args.name}_val"
                                    / "confusion_matrix.png"),
            "confusion_matrix_normalized": str(
                TRAIN_OUT / f"{args.name}_val" / "confusion_matrix_normalized.png"),
            "custom_confusion": str(FIG_OUT / "fig_task4_2_confusion.png"),
            "pr_curve": str(TRAIN_OUT / f"{args.name}_val" / "BoxPR_curve.png"),
            "f1_curve": str(TRAIN_OUT / f"{args.name}_val" / "BoxF1_curve.png"),
            "val_batch0_pred": str(TRAIN_OUT / f"{args.name}_val"
                                   / "val_batch0_pred.jpg"),
            "labels": str(run_dir / "labels.jpg"),
            "results_csv": str(run_dir / "results.csv"),
        },
    }
    save_json(STAT_OUT / "train_result.json", train_result)
    print(json.dumps(train_result, ensure_ascii=False, indent=2))
    print(f"[完成] 训练结果目录: {run_dir}")


if __name__ == "__main__":
    main()



