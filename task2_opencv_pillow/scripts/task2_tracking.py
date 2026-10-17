"""task-2 题 5：使用 OpenCV Tracker 追踪 example3.mp4 / example4.mp4 中的小球。

* example3.mp4：红色小球在纯色背景上运动，用 HSV 颜色阈值即可得到可靠的参考框；
  对比 CSRT / KCF / MOSSE 三种跟踪器，并实现“颜色检测 + CSRT + Kalman”的混合跟踪。
* example4.mp4：白色乒乓球在木桌上被手移动，镜头也在运动，且存在手部遮挡；
  以首帧手动框初始化，对比纯 CSRT 与“模板匹配重检测 + CSRT + Kalman”的混合跟踪，
  并用模板相关分数作为无标注条件下的跟踪有效性指标。

输出：
    outputs/videos/example3_{csrt,kcf,mosse,hybrid}.mp4
    outputs/videos/example4_{csrt,hybrid}.mp4
    outputs/images/fig_task2_5_tracking_example3.png
    outputs/images/fig_task2_5_tracking_example4.png
    outputs/stats/task2_tracking_stats.json

运行：
    python task2_opencv_pillow/scripts/task2_tracking.py
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

# example4 首帧中乒乓球的位置（人工框选，作为跟踪初始化）
EXAMPLE4_INIT_BOX = (1470, 810, 180, 180)


# ----------------------------- 通用工具 -----------------------------
def box_iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def clip_box(box, w, h):
    x, y, bw, bh = [int(round(v)) for v in box]
    x = max(0, min(x, w - 2))
    y = max(0, min(y, h - 2))
    bw = max(2, min(bw, w - x))
    bh = max(2, min(bh, h - y))
    return x, y, bw, bh


def create_tracker(kind: str):
    """兼容 OpenCV 4.x / 5.x 的 Tracker 构造接口。"""
    legacy = getattr(cv2, "legacy", None)
    names = {
        "csrt": ["TrackerCSRT_create", "TrackerCSRT"],
        "kcf": ["TrackerKCF_create", "TrackerKCF"],
        "mosse": ["TrackerMOSSE_create", "TrackerMOSSE"],
    }[kind]
    for owner in (cv2, legacy):
        if owner is None:
            continue
        for name in names:
            if hasattr(owner, name):
                obj = getattr(owner, name)
                try:
                    return obj()
                except TypeError:
                    return obj
    raise RuntimeError(f"当前 OpenCV 不支持 {kind} tracker，请安装 opencv-contrib-python")


def draw_status(frame, box, status, color, trail, label):
    x, y, w, h = clip_box(box, frame.shape[1], frame.shape[0])
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    pts = np.array(trail[-40:], dtype=np.int32).reshape(-1, 1, 2)
    if len(pts) > 1:
        cv2.polylines(frame, [pts], False, color, 2, cv2.LINE_AA)
    cx, cy = x + w // 2, y + h // 2
    cv2.circle(frame, (cx, cy), 4, color, -1)
    text = f"{label} | {status}"
    cv2.rectangle(frame, (8, 8), (8 + 15 * len(text), 44), (0, 0, 0), -1)
    cv2.putText(frame, text, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                cv2.LINE_AA)


# ----------------------------- 颜色检测 -----------------------------
def detect_red_ball(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 110, 60), (12, 255, 255))
    m2 = cv2.inRange(hsv, (168, 110, 60), (180, 255, 255))
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 200:
        return None
    x, y, w, h = cv2.boundingRect(c)
    pad = int(0.08 * max(w, h))
    return clip_box((x - pad, y - pad, w + 2 * pad, h + 2 * pad),
                    frame.shape[1], frame.shape[0])


def red_pixel_ratio(frame, box) -> float:
    x, y, w, h = clip_box(box, frame.shape[1], frame.shape[0])
    hsv = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2HSV)
    if hsv.size == 0:
        return 0.0
    m = (((hsv[:, :, 0] < 12) | (hsv[:, :, 0] > 168)) & (hsv[:, :, 1] > 110)
         & (hsv[:, :, 2] > 60))
    return float(m.mean())


# ----------------------------- 模板匹配 -----------------------------
def template_detect(frame, template, center, last_scale=1.0,
                    search_radius=160):
    """在预测中心附近做多尺度模板匹配，并结合“蓝白色”约束抑制误匹配。"""
    h, w = frame.shape[:2]
    cx, cy = int(center[0]), int(center[1])
    r = search_radius
    x0, y0 = max(0, cx - r), max(0, cy - r)
    x1, y1 = min(w, cx + r), min(h, cy + r)
    patch = frame[y0:y1, x0:x1]
    if patch.size == 0:
        return None, 0.0, 0.0
    best = (-1.0, None, 0.0, 0.0)  # (adj_score, box, raw_score, color_frac)
    th0, tw0 = template.shape[:2]
    lo = max(0.25, last_scale * 0.65)
    hi = min(1.45, last_scale * 1.40)
    for s in np.linspace(lo, hi, 10):
        tw, th = int(tw0 * s), int(th0 * s)
        if tw < 24 or th < 24 or tw >= patch.shape[1] or th >= patch.shape[0]:
            continue
        tmpl = cv2.resize(template, (tw, th), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(patch, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        box = (x0 + max_loc[0], y0 + max_loc[1], tw, th)
        color_frac = blue_white_fraction(frame, box)
        # 相关系数 + 颜色先验 - 尺度突变惩罚
        adj = max_val + 0.25 * color_frac - 0.08 * abs(np.log(max(s, 1e-3)
                                                             / max(last_scale, 1e-3)))
        if adj > best[0]:
            best = (adj, box, float(max_val), color_frac)
    return best[1], best[2], best[3]


def blue_white_fraction(frame, box) -> float:
    """目标区域的蓝白色像素比例：乒乓球在暖色木桌上呈偏蓝的白色。"""
    x, y, w, h = clip_box(box, frame.shape[1], frame.shape[0])
    crop = frame[y:y + h, x:x + w]
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 0] >= 70) & (hsv[:, :, 0] <= 140)
            & (hsv[:, :, 1] < 110) & (hsv[:, :, 2] > 100))
    return float(mask.mean())


def appearance_score(frame, template, box) -> float:
    """把模板缩放到跟踪框大小，计算外观相关分数（0~1）。"""
    x, y, w, h = clip_box(box, frame.shape[1], frame.shape[0])
    if w < 8 or h < 8:
        return 0.0
    crop = frame[y:y + h, x:x + w]
    tmpl = cv2.resize(template, (w, h), interpolation=cv2.INTER_AREA)
    if crop.shape[:2] != tmpl.shape[:2]:
        return 0.0
    res = cv2.matchTemplate(crop, tmpl, cv2.TM_CCOEFF_NORMED)
    return float(res[0, 0])


class Kalman2D:
    def __init__(self, center):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]],
                                             np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                                             [0, 0, 1, 0], [0, 0, 0, 1]],
                                            np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5e-2
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.kf.statePost = np.array([[center[0]], [center[1]], [0], [0]],
                                     np.float32)

    def predict(self):
        s = self.kf.predict()
        return float(s[0, 0]), float(s[1, 0])

    def correct(self, center):
        self.kf.correct(np.array([[center[0]], [center[1]]], np.float32))


# ----------------------------- example3 -----------------------------
def run_example3():
    src = DATA / "example3.mp4"
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, frame0 = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(src)
    init_box = detect_red_ball(frame0)
    print(f"[example3] 初始框 {init_box}, 帧数={n}, fps={fps:.2f}")

    results = {}
    # 参考框（颜色检测）
    refs = []
    cap = cv2.VideoCapture(str(src))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        refs.append(detect_red_ball(frame))
    cap.release()

    methods = ["csrt", "kcf", "mosse"]
    for method in methods:
        cap = cv2.VideoCapture(str(src))
        ok, frame = cap.read()
        tracker = create_tracker(method)
        tracker.init(frame, init_box)
        writer = cv2.VideoWriter(str(VID_OUT / f"example3_{method}.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                 (frame.shape[1], frame.shape[0]))
        trail, ious, centers, valid, times = [], [], [], [], []
        last_box = init_box
        lost = 0
        idx = 0
        while ok:
            t0 = time.perf_counter()
            if idx > 0:
                ok, frame = cap.read()
                if not ok:
                    break
            ok_track, box = tracker.update(frame)
            if not ok_track:
                box = last_box
                lost += 1
            last_box = box
            box = clip_box(box, frame.shape[1], frame.shape[0])
            cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
            trail.append((int(cx), int(cy)))
            centers.append((cx, cy))
            ref = refs[idx] if idx < len(refs) else None
            iou = box_iou(box, ref) if ref else 0.0
            ious.append(iou)
            valid.append(iou >= 0.5)
            times.append(time.perf_counter() - t0)
            draw_status(frame, box, "TRACK" if ok_track else "LOST",
                        (0, 255, 0) if ok_track else (0, 0, 255),
                        trail, method.upper())
            writer.write(frame)
            idx += 1
        cap.release()
        writer.release()
        elapsed = sum(times)
        det_vals = [iou for iou in ious]
        results[method] = {
            "frames": len(ious),
            "success_rate": round(float(np.mean(valid)), 4),
            "mean_iou": round(float(np.mean(det_vals)), 4),
            "tracker_lost_frames": lost,
            "fps": round(len(ious) / max(elapsed, 1e-9), 2),
            "centers": centers,
            "ious": ious,
        }
        print(f"[example3] {method.upper()} success={results[method]['success_rate']:.3f} "
              f"mIoU={results[method]['mean_iou']:.3f} fps={results[method]['fps']:.1f}")

    # 混合：CSRT + 颜色重检测 + Kalman
    cap = cv2.VideoCapture(str(src))
    ok, frame = cap.read()
    tracker = create_tracker("csrt")
    tracker.init(frame, init_box)
    c0 = (init_box[0] + init_box[2] / 2, init_box[1] + init_box[3] / 2)
    kf = Kalman2D(c0)
    writer = cv2.VideoWriter(str(VID_OUT / "example3_hybrid.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (frame.shape[1], frame.shape[0]))
    trail, ious, centers, reinit_count, times = [], [], [], 0, []
    last_box = init_box
    idx = 0
    while ok:
        t0 = time.perf_counter()
        if idx > 0:
            ok, frame = cap.read()
            if not ok:
                break
        pred = kf.predict()
        ok_track, box = tracker.update(frame)
        if not ok_track:
            box = last_box
        det = detect_red_ball(frame)
        if det is not None:
            dc = (det[0] + det[2] / 2, det[1] + det[3] / 2)
            kf.correct(dc)
            # 跟踪框与检测框偏差较大时，用检测框重置跟踪器
            if (not ok_track) or box_iou(box, det) < 0.35:
                tracker = create_tracker("csrt")
                tracker.init(frame, det)
                box = det
                reinit_count += 1
            else:
                # 用检测框中心做 Kalman 平滑，再保持跟踪框尺寸
                cx, cy = dc
                box = (cx - box[2] / 2, cy - box[3] / 2, box[2], box[3])
        else:
            # 目标丢失：使用 Kalman 预测位置，保持上一帧尺寸
            box = (pred[0] - box[2] / 2, pred[1] - box[3] / 2, box[2], box[3])
        box = clip_box(box, frame.shape[1], frame.shape[0])
        last_box = box
        cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
        trail.append((int(cx), int(cy)))
        centers.append((cx, cy))
        ref = refs[idx] if idx < len(refs) else None
        iou = box_iou(box, ref) if ref else 0.0
        ious.append(iou)
        times.append(time.perf_counter() - t0)
        draw_status(frame, box, "HYBRID", (255, 128, 0), trail, "HYBRID")
        writer.write(frame)
        idx += 1
    cap.release()
    writer.release()
    elapsed = sum(times)
    results["hybrid"] = {
        "frames": len(ious),
        "success_rate": round(float(np.mean([i >= 0.5 for i in ious])), 4),
        "mean_iou": round(float(np.mean(ious)), 4),
        "reinit_count": reinit_count,
        "fps": round(len(ious) / max(elapsed, 1e-9), 2),
        "centers": centers,
        "ious": ious,
    }
    print(f"[example3] HYBRID success={results['hybrid']['success_rate']:.3f} "
          f"mIoU={results['hybrid']['mean_iou']:.3f} "
          f"reinit={reinit_count} fps={results['hybrid']['fps']:.1f}")

    # 轨迹图 + IoU 曲线
    cap = cv2.VideoCapture(str(src))
    ok, bg = cap.read()
    cap.release()
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    bg_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
    axes[0].imshow(bg_rgb)
    colors = {"csrt": "#1f77b4", "kcf": "#2ca02c", "mosse": "#9467bd",
              "hybrid": "#ff7f0e"}
    for m, r in results.items():
        pts = np.array(r["centers"], dtype=float)
        axes[0].plot(pts[:, 0], pts[:, 1], lw=1.6, label=m.upper(),
                     color=colors[m])
    axes[0].set_xlim(0, bg.shape[1])
    axes[0].set_ylim(bg.shape[0], 0)
    axes[0].set_title("跟踪中心轨迹（example3，红色小球）")
    axes[0].legend(fontsize=9)
    axes[0].axis("off")
    for m, r in results.items():
        axes[1].plot(r["ious"], lw=1.2, label=f"{m.upper()} (成功率 {r['success_rate']:.2%})",
                     color=colors[m])
    axes[1].axhline(0.5, ls="--", c="gray", lw=1, label="IoU=0.5 判定阈值")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("帧序号")
    axes[1].set_ylabel("IoU（与颜色检测参考框）")
    axes[1].set_title("逐帧跟踪 IoU")
    axes[1].legend(fontsize=8, loc="lower left")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMG_OUT / "fig_task2_5_tracking_example3.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    return {k: {kk: vv for kk, vv in v.items() if kk not in ("centers", "ious")}
            for k, v in results.items()}, results


# ----------------------------- example4 -----------------------------
def run_example4():
    src = DATA / "example4.mp4"
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, frame0 = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(src)
    init_box = clip_box(EXAMPLE4_INIT_BOX, frame0.shape[1], frame0.shape[0])
    template = frame0[init_box[1]:init_box[1] + init_box[3],
                      init_box[0]:init_box[0] + init_box[2]].copy()
    print(f"[example4] 初始框 {init_box}, 模板 {template.shape[:2]}, "
          f"帧数={n}, fps={fps:.2f}")

    results = {}

    # 纯 CSRT
    cap = cv2.VideoCapture(str(src))
    ok, frame = cap.read()
    tracker = create_tracker("csrt")
    tracker.init(frame, init_box)
    writer = cv2.VideoWriter(str(VID_OUT / "example4_csrt.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (frame.shape[1], frame.shape[0]))
    trail, scores, centers, times = [], [], [], []
    last_box = init_box
    lost = 0
    idx = 0
    while ok:
        t0 = time.perf_counter()
        if idx > 0:
            ok, frame = cap.read()
            if not ok:
                break
        ok_track, box = tracker.update(frame)
        if not ok_track:
            box = last_box
            lost += 1
        box = clip_box(box, frame.shape[1], frame.shape[0])
        last_box = box
        cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
        trail.append((int(cx), int(cy)))
        centers.append((cx, cy))
        scores.append(appearance_score(frame, template, box))
        times.append(time.perf_counter() - t0)
        draw_status(frame, box, "TRACK" if ok_track else "LOST",
                    (0, 255, 0) if ok_track else (0, 0, 255), trail, "CSRT")
        writer.write(frame)
        idx += 1
    cap.release()
    writer.release()
    elapsed = sum(times)
    results["csrt"] = {
        "frames": len(scores),
        "valid_rate": round(float(np.mean([s >= 0.45 for s in scores])), 4),
        "mean_appearance_score": round(float(np.mean(scores)), 4),
        "tracker_lost_frames": lost,
        "fps": round(len(scores) / max(elapsed, 1e-9), 2),
        "centers": centers,
        "scores": scores,
    }
    print(f"[example4] CSRT valid={results['csrt']['valid_rate']:.3f} "
          f"score={results['csrt']['mean_appearance_score']:.3f} "
          f"fps={results['csrt']['fps']:.1f}")

    # 混合：CSRT + 模板匹配重检测 + Kalman
    cap = cv2.VideoCapture(str(src))
    ok, frame = cap.read()
    tracker = create_tracker("csrt")
    tracker.init(frame, init_box)
    c0 = (init_box[0] + init_box[2] / 2, init_box[1] + init_box[3] / 2)
    kf = Kalman2D(c0)
    writer = cv2.VideoWriter(str(VID_OUT / "example4_hybrid.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (frame.shape[1], frame.shape[0]))
    trail, scores, centers, times = [], [], [], []
    reinit_count, occluded = 0, 0
    detect_frames, track_frames = 0, 0
    idx = 0
    last_box = init_box
    last_scale = 1.0
    miss = 0
    while ok:
        t0 = time.perf_counter()
        if idx > 0:
            ok, frame = cap.read()
            if not ok:
                break
        pred = kf.predict()
        ok_track, box = tracker.update(frame)
        if not ok_track:
            box = last_box
        # 连续丢失时扩大搜索范围
        radius = 160 if miss < 6 else 320
        det, raw, color_frac = template_detect(frame, template, pred,
                                               last_scale, radius)
        valid_det = (det is not None and raw >= 0.55 and color_frac >= 0.45)
        if valid_det:
            dc = (det[0] + det[2] / 2, det[1] + det[3] / 2)
            if ok_track and box_iou(box, det) >= 0.15:
                # 跟踪器仍可靠：优先使用跟踪框，检测框只用于修正 Kalman
                bc = (box[0] + box[2] / 2, box[1] + box[3] / 2)
                kf.correct(bc)
                status = "TRACK"
                color = (0, 255, 0)
                track_frames += 1
                miss = 0
            else:
                kf.correct(dc)
                tracker = create_tracker("csrt")
                tracker.init(frame, det)
                reinit_count += 1
                box = det
                last_scale = det[2] / template.shape[1]
                status = "DETECT"
                color = (255, 128, 0)
                detect_frames += 1
                miss = 0
        else:
            # 未检测到：若跟踪框本身仍与模板相似则继续使用，否则用 Kalman 预测
            t_score = appearance_score(frame, template, box) if ok_track else 0.0
            if ok_track and t_score >= 0.45:
                bc = (box[0] + box[2] / 2, box[1] + box[3] / 2)
                kf.correct(bc)
                status = "TRACK"
                color = (0, 255, 0)
                track_frames += 1
                miss = 0
            else:
                box = (pred[0] - last_box[2] / 2, pred[1] - last_box[3] / 2,
                       last_box[2], last_box[3])
                status = "PREDICT"
                color = (0, 0, 255)
                occluded += 1
                miss += 1
        box = clip_box(box, frame.shape[1], frame.shape[0])
        last_box = box
        cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
        trail.append((int(cx), int(cy)))
        centers.append((cx, cy))
        scores.append(appearance_score(frame, template, box))
        times.append(time.perf_counter() - t0)
        draw_status(frame, box, status, color, trail, "HYBRID")
        writer.write(frame)
        idx += 1
    cap.release()
    writer.release()
    elapsed = sum(times)
    results["hybrid"] = {
        "frames": len(scores),
        "valid_rate": round(float(np.mean([s >= 0.45 for s in scores])), 4),
        "mean_appearance_score": round(float(np.mean(scores)), 4),
        "reinit_count": reinit_count,
        "predict_frames": occluded,
        "track_frames": track_frames,
        "detect_frames": detect_frames,
        "fps": round(len(scores) / max(elapsed, 1e-9), 2),
        "centers": centers,
        "scores": scores,
    }
    print(f"[example4] HYBRID valid={results['hybrid']['valid_rate']:.3f} "
          f"score={results['hybrid']['mean_appearance_score']:.3f} "
          f"reinit={reinit_count} fps={results['hybrid']['fps']:.1f}")

    # 轨迹图 + 外观分数曲线
    cap = cv2.VideoCapture(str(src))
    ok, bg = cap.read()
    cap.release()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    axes[0].imshow(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
    for m, color in (("csrt", "#1f77b4"), ("hybrid", "#ff7f0e")):
        pts = np.array(results[m]["centers"], dtype=float)[::2]
        if len(pts) > 1:
            d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            x, y = pts[:, 0].copy(), pts[:, 1].copy()
            jump = d > 150
            x[1:][jump] = np.nan
            y[1:][jump] = np.nan
            axes[0].plot(x, y, lw=1.4, label=m.upper(), color=color)
    axes[0].set_xlim(0, bg.shape[1])
    axes[0].set_ylim(bg.shape[0], 0)
    axes[0].set_title("跟踪中心轨迹（example4，白色乒乓球）")
    axes[0].legend(fontsize=9)
    axes[0].axis("off")
    for m, color in (("csrt", "#1f77b4"), ("hybrid", "#ff7f0e")):
        axes[1].plot(results[m]["scores"], lw=1.0, alpha=0.85,
                     label=f"{m.upper()} (有效率 {results[m]['valid_rate']:.2%})",
                     color=color)
    axes[1].axhline(0.45, ls="--", c="gray", lw=1, label="有效性阈值 0.45")
    axes[1].set_ylim(-0.1, 1.0)
    axes[1].set_xlabel("帧序号")
    axes[1].set_ylabel("模板外观相关分数")
    axes[1].set_title("逐帧跟踪有效性（无真值视频的替代指标）")
    axes[1].legend(fontsize=8, loc="lower left")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMG_OUT / "fig_task2_5_tracking_example4.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    return {k: {kk: vv for kk, vv in v.items() if kk not in ("centers", "scores")}
            for k, v in results.items()}, results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["all", "example3", "example4"],
                        default="all")
    args = parser.parse_args()
    for d in (VID_OUT, IMG_OUT, STAT_OUT):
        d.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    stats_file = STAT_OUT / "task2_tracking_stats.json"
    old = json.loads(stats_file.read_text(encoding="utf-8")) \
        if stats_file.exists() else {}
    stats3 = old.get("example3")
    stats4 = old.get("example4")
    if args.only in ("all", "example3"):
        stats3, _ = run_example3()
    if args.only in ("all", "example4"):
        stats4, _ = run_example4()
    stats = {
        "example3": stats3,
        "example4": stats4,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "notes": {
            "example3_reference": "HSV 红色阈值检测框作为参考（真值代理）",
            "example4_reference": "模板外观相关分数 >= 0.45 视为跟踪有效（无真值）",
            "example4_init_box": EXAMPLE4_INIT_BOX,
        },
    }
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"[完成] 视频目录: {VID_OUT}")


if __name__ == "__main__":
    main()
