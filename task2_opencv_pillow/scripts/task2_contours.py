"""task-2 题 3：检测 example1.jpg 中的目标轮廓，绘制并统计总面积 / 总周长。

思路：
1. 飞机为白色/深蓝色机身，天空为高饱和蓝色。利用 B-R 色差把天空作为背景剔除，
   得到前景掩膜（不依赖人工框选）；
2. 限定在目标所在的水平带内，形态学闭运算连接机身、机翼、尾翼等部件；
3. 取最大连通域并填充内部孔洞，提取外轮廓；
4. 计算轮廓面积、周长（以及凸包 / 多边形近似的对比数据），绘制并保存结果。

运行：
    python task2_opencv_pillow/scripts/task2_contours.py
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "task2"
OUT = ROOT / "task2_opencv_pillow" / "outputs"
IMG_OUT = OUT / "images"
STAT_OUT = OUT / "stats"

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def imread_unicode(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, flags)
    if img is None:
        raise FileNotFoundError(path)
    return img


def imwrite_unicode(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, img)
    if not ok:
        raise RuntimeError(path)
    buf.tofile(str(path))


def build_plane_mask(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """利用“天空为蓝色、飞机为低饱和白/深蓝”的颜色先验分割飞机。"""
    h, w = img.shape[:2]
    b, g, r = cv2.split(img.astype(np.int16))
    # 天空：蓝色分量明显高于红色，且足够亮
    sky = ((b - r > 25) & (b > 110) & (g > 70)).astype(np.uint8) * 255
    plane = cv2.bitwise_not(sky)

    # 只保留目标所在的水平带，排除上下边缘的云层
    plane[: int(h * 0.22), :] = 0
    plane[int(h * 0.80):, :] = 0

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    plane = cv2.morphologyEx(plane, cv2.MORPH_CLOSE, k_close)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    plane = cv2.morphologyEx(plane, cv2.MORPH_OPEN, k_open)

    # 取最大连通域（飞机），再闭运算 + 填孔，得到完整剪影
    n, labels, stats, _ = cv2.connectedComponentsWithStats(plane, connectivity=8)
    if n <= 1:
        raise RuntimeError("未分割出前景目标")
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == idx).astype(np.uint8) * 255

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [cnt], -1, 255, thickness=cv2.FILLED)
    return filled, cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 60, 160)


def main() -> None:
    IMG_OUT.mkdir(parents=True, exist_ok=True)
    STAT_OUT.mkdir(parents=True, exist_ok=True)

    img = imread_unicode(DATA / "example1.jpg")
    h, w = img.shape[:2]
    mask, edges = build_plane_mask(img)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    main_cnt = contours[0]

    area = float(cv2.contourArea(main_cnt))
    perimeter = float(cv2.arcLength(main_cnt, True))

    hull = cv2.convexHull(main_cnt)
    hull_area = float(cv2.contourArea(hull))
    hull_perimeter = float(cv2.arcLength(hull, True))

    epsilon = 0.01 * perimeter
    approx = cv2.approxPolyDP(main_cnt, epsilon, True)
    approx_area = float(cv2.contourArea(approx))
    approx_perimeter = float(cv2.arcLength(approx, True))

    x, y, bw, bh = cv2.boundingRect(main_cnt)
    moments = cv2.moments(main_cnt)
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    # 所有较显著轮廓的统计（用于说明“总”面积/周长）
    all_cnts = [c for c in contours if cv2.contourArea(c) >= 200]
    total_area_all = float(sum(cv2.contourArea(c) for c in all_cnts))
    total_perimeter_all = float(sum(cv2.arcLength(c, True) for c in all_cnts))

    stats = {
        "image_size": [w, h],
        "image_area_px": w * h,
        "num_contours_all": len(contours),
        "num_contours_significant": len(all_cnts),
        "main_contour": {
            "area_px": round(area, 2),
            "perimeter_px": round(perimeter, 2),
            "area_ratio_percent": round(area / (w * h) * 100, 3),
            "bbox_xywh": [int(x), int(y), int(bw), int(bh)],
            "centroid_xy": [round(cx, 2), round(cy, 2)],
            "num_points": int(len(main_cnt)),
        },
        "convex_hull": {
            "area_px": round(hull_area, 2),
            "perimeter_px": round(hull_perimeter, 2),
            "num_points": int(len(hull)),
        },
        "polygon_approx_1pct": {
            "epsilon_px": round(epsilon, 2),
            "area_px": round(approx_area, 2),
            "perimeter_px": round(approx_perimeter, 2),
            "num_vertices": int(len(approx)),
        },
        "sum_significant_contours": {
            "area_px": round(total_area_all, 2),
            "perimeter_px": round(total_perimeter_all, 2),
        },
    }

    with open(STAT_OUT / "task2_contour_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 保存掩膜和标注图
    imwrite_unicode(IMG_OUT / "task2_contour_mask.png", mask)
    overlay = img.copy()
    cv2.drawContours(overlay, [main_cnt], -1, (0, 0, 255), 3)
    cv2.drawContours(overlay, [hull], -1, (0, 255, 255), 2)
    cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
    cv2.circle(overlay, (int(cx), int(cy)), 5, (255, 0, 255), -1)
    label = f"Area={area:.0f}px  Perimeter={perimeter:.0f}px"
    cv2.putText(overlay, label, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 255), 2, cv2.LINE_AA)
    imwrite_unicode(IMG_OUT / "task2_contour_overlay.png", overlay)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("原图 example1.jpg", fontsize=12)
    axes[0, 1].imshow(mask, cmap="gray")
    axes[0, 1].set_title("颜色先验 + 形态学得到的目标掩膜", fontsize=12)
    axes[1, 0].imshow(edges, cmap="gray")
    axes[1, 0].set_title("Canny 边缘（辅助观察）", fontsize=12)
    axes[1, 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(
        f"外轮廓（红） 凸包（黄） 面积={area:.0f}px^2 周长={perimeter:.0f}px",
        fontsize=12)
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(IMG_OUT / "fig_task2_3_contours.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"[完成] 轮廓图: {IMG_OUT / 'task2_contour_overlay.png'}")


if __name__ == "__main__":
    main()
