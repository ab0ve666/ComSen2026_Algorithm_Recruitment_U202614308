"""task-2 题 1、题 2：OpenCV / Pillow 读图、显示、通道分离与几何变换。

运行：
    python task2_opencv_pillow/scripts/task2_image_ops.py

输出：
    task2_opencv_pillow/outputs/images/fig_task2_1_channels.png
    task2_opencv_pillow/outputs/images/fig_task2_2_transforms.png
    task2_opencv_pillow/outputs/images/transforms/*.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "task2"
OUT = ROOT / "task2_opencv_pillow" / "outputs"
IMG_OUT = OUT / "images"
TRANS_OUT = IMG_OUT / "transforms"

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def imread_unicode(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """cv2.imread 在中文路径下会失败，用 imdecode 读取。"""
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, flags)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return img


def imwrite_unicode(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, img)
    if not ok:
        raise RuntimeError(f"无法编码图像: {path}")
    buf.tofile(str(path))


def show_panel(ax, img_rgb: np.ndarray, title: str) -> None:
    ax.imshow(img_rgb)
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def colorize_channel(ch: np.ndarray, index: int) -> np.ndarray:
    """把单通道放到 B/G/R 对应位置，其余通道置零，得到彩色可视化。"""
    zeros = np.zeros_like(ch)
    bgr = [zeros, zeros, zeros]
    bgr[index] = ch
    return cv2.merge(bgr)


def contact_sheet(items, cols: int = 4, tile=(480, 300), label_h: int = 34):
    """PIL 拼接等尺寸网格，避免旋转后不同长宽比导致的排版错位。"""
    from PIL import ImageDraw, ImageFont

    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 20)
    tw, th = tile
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * th), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (label, bgr) in enumerate(items):
        r, c = divmod(i, cols)
        ox, oy = c * tw, r * th
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        max_w, max_h = tw - 20, th - label_h - 12
        scale = min(max_w / im.width, max_h / im.height, 1.0)
        im = im.resize((max(1, int(im.width * scale)),
                        max(1, int(im.height * scale))),
                       Image.Resampling.LANCZOS)
        # 标签居中
        bbox = draw.textbbox((0, 0), label, font=font)
        draw.text((ox + (tw - (bbox[2] - bbox[0])) / 2, oy + 6),
                  label, fill="black", font=font)
        sheet.paste(im, (ox + (tw - im.width) // 2,
                         oy + label_h + (max_h - im.height) // 2))
    return np.array(sheet)


def main() -> None:
    IMG_OUT.mkdir(parents=True, exist_ok=True)
    TRANS_OUT.mkdir(parents=True, exist_ok=True)

    img_path = DATA / "example1.jpg"
    img_cv = imread_unicode(img_path)          # OpenCV: BGR
    img_pil = np.array(Image.open(img_path).convert("RGB"))  # Pillow: RGB

    print(f"[读取] OpenCV shape={img_cv.shape}, dtype={img_cv.dtype}")
    print(f"[读取] Pillow shape={img_pil.shape}, dtype={img_pil.dtype}")
    print("[校验] OpenCV(BGR)->RGB 与 Pillow(RGB) 是否逐像素一致:",
          np.array_equal(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), img_pil))

    # Pillow 显示（保存为文件，服务器无图形界面时也能查看）
    pil_dir = IMG_OUT / "display"
    pil_dir.mkdir(parents=True, exist_ok=True)
    img_pil_obj = Image.open(img_path).convert("RGB")
    img_pil_obj.save(pil_dir / "example1_pillow_display.png")
    # OpenCV 显示：无 GUI 环境下用 imencode 落盘代替 imshow
    imwrite_unicode(pil_dir / "example1_opencv_display.png", img_cv)

    # ---------- 题 1：通道分离 ----------
    b, g, r = cv2.split(img_cv)  # OpenCV 顺序是 B,G,R
    r_pil, g_pil, b_pil = img_pil_obj.split()  # Pillow 顺序是 R,G,B

    for name, ch in (("R", r), ("G", g), ("B", b)):
        imwrite_unicode(IMG_OUT / f"channel_{name}.png", ch)
    # 验证两库通道一致
    assert np.array_equal(r, np.array(r_pil))
    assert np.array_equal(g, np.array(g_pil))
    assert np.array_equal(b, np.array(b_pil))

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    show_panel(axes[0, 0], bgr_to_rgb(img_cv), "原图（OpenCV BGR→RGB）")
    for i, (name, ch) in enumerate((("R", r), ("G", g), ("B", b))):
        show_panel(axes[0, i + 1], ch, f"{name} 通道（灰度）")
    show_panel(axes[1, 0], bgr_to_rgb(img_cv), "原图（Pillow RGB）")
    for i, (name, ch, idx) in enumerate((("R", r, 2), ("G", g, 1), ("B", b, 0))):
        show_panel(axes[1, i + 1], bgr_to_rgb(colorize_channel(ch, idx)),
                   f"{name} 通道（彩色显示）")
    fig.suptitle("题 1：读取显示与 B/G/R 三通道分离", fontsize=15)
    fig.tight_layout()
    fig.savefig(IMG_OUT / "fig_task2_1_channels.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------- 题 2：缩放、旋转、平移、翻转 ----------
    h, w = img_cv.shape[:2]
    center = (w / 2, h / 2)

    resized_small = cv2.resize(img_cv, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    resized_big = cv2.resize(img_cv, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    def rotate_expand(image: np.ndarray, angle: float) -> np.ndarray:
        """绕中心旋转并扩大画布，避免四角被裁掉。"""
        ih, iw = image.shape[:2]
        c = (iw / 2, ih / 2)
        m = cv2.getRotationMatrix2D(c, angle, 1.0)
        cos, sin = abs(m[0, 0]), abs(m[0, 1])
        nw, nh = int(ih * sin + iw * cos), int(ih * cos + iw * sin)
        m[0, 2] += nw / 2 - c[0]
        m[1, 2] += nh / 2 - c[1]
        return cv2.warpAffine(image, m, (nw, nh), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    rotated_45 = rotate_expand(img_cv, 45)
    rotated_m30 = rotate_expand(img_cv, -30)

    tx, ty = 120, 60
    translated = cv2.warpAffine(
        img_cv, np.float32([[1, 0, tx], [0, 1, ty]]), (w, h),
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    flip_h = cv2.flip(img_cv, 1)
    flip_v = cv2.flip(img_cv, 0)

    # Pillow 对应实现（交叉验证两种库的结果）
    pil_small = img_pil_obj.resize((w // 2, h // 2), Image.Resampling.LANCZOS)
    pil_rot = img_pil_obj.rotate(45, expand=True, fillcolor=(255, 255, 255))

    for name, im in (("resize_0.5x", resized_small), ("resize_1.5x", resized_big),
                     ("rotate_45", rotated_45), ("rotate_-30", rotated_m30),
                     ("translate_120_60", translated), ("flip_horizontal", flip_h),
                     ("flip_vertical", flip_v)):
        imwrite_unicode(TRANS_OUT / f"{name}.png", im)

    sheet = contact_sheet([
        (f"原图 {w}×{h}", img_cv),
        (f"缩放 0.5×  {resized_small.shape[1]}×{resized_small.shape[0]}",
         resized_small),
        (f"缩放 1.5×  {resized_big.shape[1]}×{resized_big.shape[0]}",
         resized_big),
        (f"旋转 +45°  {rotated_45.shape[1]}×{rotated_45.shape[0]}", rotated_45),
        (f"旋转 -30°  {rotated_m30.shape[1]}×{rotated_m30.shape[0]}", rotated_m30),
        (f"平移 tx={tx}, ty={ty}", translated),
        ("水平翻转（左右镜像）", flip_h),
        ("垂直翻转（上下镜像）", flip_v),
    ], cols=4, tile=(500, 330))
    fig, ax = plt.subplots(figsize=(20, 6.6))
    ax.imshow(sheet)
    ax.set_title("题 2：图像缩放 / 旋转 / 平移 / 翻转", fontsize=15)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(IMG_OUT / "fig_task2_2_transforms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[完成] 输出目录: {IMG_OUT}")
    print(f"[核对] 0.5× 尺寸 OpenCV={resized_small.shape[:2][::-1]}, "
          f"Pillow={pil_small.size}")
    print(f"[核对] 旋转图像尺寸 OpenCV={rotated_45.shape[:2][::-1]}, "
          f"Pillow={pil_rot.size}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"[错误] {exc}", file=sys.stderr)
        raise
