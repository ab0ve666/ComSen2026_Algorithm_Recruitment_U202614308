"""task-4 题目一（5，提高部分）：模型导出与 FP16 / INT8 量化，比较大小、精度与速度。

流程：
1. 将 best.pt 导出为 ONNX（FP32）；
2. 导出 FP16 ONNX；
3. 用验证集图片做校准，进行 ONNX Runtime 静态 INT8 量化；
4. 在验证集上评估 mAP（FP32 / FP16 / INT8），并基准测试纯推理延迟；
5. 输出对比表与柱状图。

运行：
    python task4_yolo_drone/src/04_export_quantize.py
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# 让 onnxruntime-gpu 能找到 PyTorch 自带的 CUDA / cuDNN DLL
_torch_lib = Path(torch.__file__).parent / "lib"
if hasattr(os, "add_dll_directory") and _torch_lib.exists():
    try:
        os.add_dll_directory(str(_torch_lib))
    except OSError:
        pass

import onnxruntime as ort
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod,
                                      QuantFormat, QuantType, quantize_dynamic,
                                      quantize_static)
from ultralytics import YOLO

from common import (FIG_OUT, MODEL_OUT, STAT_OUT, file_size_mb,
                    find_best_weights, find_data_yaml, load_yaml, save_json,
                    setup_matplotlib)


# ----------------------------- 预处理 -----------------------------
def letterbox(img: np.ndarray, new_shape: int = 640,
              color=(114, 114, 114)) -> np.ndarray:
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new_shape - nh) // 2
    bottom = new_shape - nh - top
    left = (new_shape - nw) // 2
    right = new_shape - nw - left
    return cv2.copyMakeBorder(img, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=color)


def preprocess(img: np.ndarray, imgsz: int = 640) -> np.ndarray:
    img = letterbox(img, imgsz)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return np.ascontiguousarray(img.transpose(2, 0, 1)[None])


class YoloCalibReader(CalibrationDataReader):
    def __init__(self, files, input_name: str, imgsz: int = 640,
                 limit: int = 150):
        self.files = list(files)[:limit]
        self.input_name = input_name
        self.imgsz = imgsz
        self.iter = iter(self.files)

    def get_next(self):
        try:
            f = next(self.iter)
        except StopIteration:
            return None
        img = cv2.imdecode(np.fromfile(str(f), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return self.get_next()
        return {self.input_name: preprocess(img, self.imgsz)}


def find_output_concat(onnx_path: Path) -> str | None:
    """找到产生最终输出 output0 的 Concat 节点名（量化时需排除，保留浮点类别分数）。"""
    try:
        import onnx

        model = onnx.load(str(onnx_path))
        graph = model.graph
        out_name = graph.output[0].name
        producer = {o: node for node in graph.node for o in node.output}
        node = producer.get(out_name)
        if node is not None and node.op_type == "Concat":
            return node.name
    except Exception as exc:
        print(f"[提示] 无法自动定位输出 Concat：{exc}")
    return None


def collect_val_images(data_yaml: Path, limit: int = 200) -> list[Path]:
    cfg = load_yaml(data_yaml)
    root = data_yaml.parent
    # Roboflow 的 data.yaml 常写 "../valid/images"，直接拼接会指到数据集外；
    # 优先使用数据集根目录下真实存在的 valid/images。
    direct = root / "valid" / "images"
    if direct.exists():
        p = direct
    else:
        val = cfg.get("val", "valid/images")
        p = Path(val)
        if not p.is_absolute():
            p = (root / p).resolve()
        if not p.exists():
            p = (root / Path(str(val).replace("../", ""))).resolve()
    if p.is_file() and p.suffix == ".txt":
        files = [root / line.strip() for line in p.read_text().splitlines()
                 if line.strip()]
    else:
        files = sorted([q for q in p.glob("*.*")
                        if q.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    return files[:limit]


# ----------------------------- 评估 / 基准 -----------------------------
def val_metrics(weights: Path, data_yaml: Path, split: str, device: str,
                half: bool = False) -> dict | None:
    try:
        model = YOLO(str(weights))
        metrics = model.val(data=str(data_yaml), split=split, imgsz=640,
                            batch=1, device=device, half=half, verbose=False,
                            plots=False)
        return {
            "mAP50": round(float(metrics.box.map50), 4),
            "mAP50_95": round(float(metrics.box.map), 4),
            "precision": round(float(metrics.box.mp), 4),
            "recall": round(float(metrics.box.mr), 4),
        }
    except Exception as exc:  # ONNX FP16 在 CPU 上可能不支持
        print(f"[评估失败] {weights.name} ({device}, half={half}): {exc}")
        return None


def bench_torch(weights: Path, device: str, half: bool, n: int = 60) -> dict:
    model = YOLO(str(weights))
    net = model.model.to(device).eval()
    if half:
        net = net.half()
    x = torch.randn(1, 3, 640, 640, device=device)
    if half:
        x = x.half()
    with torch.no_grad():
        for _ in range(15):
            net(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            net(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    ms = elapsed / n * 1000
    return {"backend": f"PyTorch {device}", "half": half,
            "latency_ms": round(ms, 3), "fps": round(1000 / ms, 2),
            "providers": None}


def bench_onnx(path: Path, half: bool, prefer_gpu: bool = False,
               n: int = 60) -> dict:
    providers = ["CPUExecutionProvider"]
    if prefer_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(path), so, providers=providers)
    input_meta = sess.get_inputs()[0]
    # Ultralytics 导出的 FP16 ONNX 输入仍是 float32，权重/计算为 FP16
    dtype = np.float16 if "float16" in input_meta.type else np.float32
    x = np.random.randn(1, 3, 640, 640).astype(dtype)
    for _ in range(15):
        sess.run(None, {input_meta.name: x})
    t0 = time.perf_counter()
    for _ in range(n):
        sess.run(None, {input_meta.name: x})
    elapsed = time.perf_counter() - t0
    ms = elapsed / n * 1000
    return {"backend": "ONNX Runtime", "half": half,
            "latency_ms": round(ms, 3), "fps": round(1000 / ms, 2),
            "providers": sess.get_providers(),
            "input_type": input_meta.type}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="")
    parser.add_argument("--calib", type=int, default=120)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--skip-val", action="store_true")
    args = parser.parse_args()

    best = Path(args.weights) if args.weights else find_best_weights()
    data_yaml = find_data_yaml()
    print(f"[权重] {best}")
    print(f"[数据] {data_yaml}")

    model = YOLO(str(best))
    fp32_onnx = MODEL_OUT / "drone_fp32.onnx"
    fp16_onnx = MODEL_OUT / "drone_fp16.onnx"
    int8_onnx = MODEL_OUT / "drone_int8.onnx"

    # 1) FP32 ONNX
    p = Path(model.export(format="onnx", imgsz=640, half=False, simplify=True,
                          opset=13, dynamic=False))
    shutil.copy2(p, fp32_onnx)
    print(f"[导出] FP32 ONNX -> {fp32_onnx} ({file_size_mb(fp32_onnx)} MB)")

    # 2) FP16 ONNX
    p = Path(model.export(format="onnx", imgsz=640, half=True, simplify=True,
                          opset=13, dynamic=False))
    shutil.copy2(p, fp16_onnx)
    print(f"[导出] FP16 ONNX -> {fp16_onnx} ({file_size_mb(fp16_onnx)} MB)")

    # 3) INT8 静态量化
    try:
        files = collect_val_images(data_yaml, limit=args.calib)
        print(f"[校准] 使用 {len(files)} 张验证集图片")
        sess = ort.InferenceSession(str(fp32_onnx),
                                    providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        reader = YoloCalibReader(files, input_name, 640, args.calib)
        concat_name = find_output_concat(fp32_onnx)
        exclude_nodes = [concat_name] if concat_name else []
        print(f"[量化] 排除输出敏感节点: {exclude_nodes}")
        quantize_static(
            model_input=str(fp32_onnx),
            model_output=str(int8_onnx),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            nodes_to_exclude=exclude_nodes,
            reduce_range=False,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
        )
        method = "static QDQ (per-channel, MinMax, 排除输出 Concat)"
        print(f"[量化] INT8 ONNX -> {int8_onnx} ({file_size_mb(int8_onnx)} MB)")
    except Exception as exc:
        print(f"[量化失败] 静态量化异常，尝试动态量化: {exc}")
        quantize_dynamic(str(fp32_onnx), str(int8_onnx),
                         weight_type=QuantType.QUInt8)
        method = "dynamic (fallback)"
        print(f"[量化] INT8 ONNX -> {int8_onnx} ({file_size_mb(int8_onnx)} MB)")

    # 4) 精度评估
    accuracy = {}
    if not args.skip_val:
        accuracy["pytorch_fp32"] = val_metrics(best, data_yaml, "val", args.device)
        accuracy["pytorch_fp16"] = val_metrics(best, data_yaml, "val", args.device,
                                               half=True)
        accuracy["onnx_fp32"] = val_metrics(fp32_onnx, data_yaml, "val", "cpu")
        accuracy["onnx_int8"] = val_metrics(int8_onnx, data_yaml, "val", "cpu")
        accuracy["onnx_fp16"] = val_metrics(fp16_onnx, data_yaml, "val", "cpu")
        for k, v in accuracy.items():
            print(f"[精度] {k}: {v}")

    # 5) 速度基准
    bench = {}

    def try_bench(name, fn):
        try:
            bench[name] = fn()
        except Exception as exc:
            print(f"[速度基准失败] {name}: {exc}")
            bench[name] = {"backend": name, "error": str(exc),
                           "latency_ms": None, "fps": None, "providers": None}

    try_bench("pytorch_fp32_gpu", lambda: bench_torch(best, "cuda:0", False))
    try_bench("pytorch_fp16_gpu", lambda: bench_torch(best, "cuda:0", True))
    try_bench("pytorch_fp32_cpu", lambda: bench_torch(best, "cpu", False))
    try_bench("onnx_fp32_cpu", lambda: bench_onnx(fp32_onnx, False, prefer_gpu=False))
    try_bench("onnx_int8_cpu", lambda: bench_onnx(int8_onnx, False, prefer_gpu=False))
    try_bench("onnx_fp16_cpu", lambda: bench_onnx(fp16_onnx, True, prefer_gpu=False))
    try_bench("onnx_fp32_cuda_attempt", lambda: bench_onnx(fp32_onnx, False, prefer_gpu=True))
    for k, v in bench.items():
        print(f"[速度] {k}: {v.get('latency_ms')} ms, {v.get('fps')} FPS "
              f"({v.get('providers')})")

    sizes = {
        "pytorch_best_pt": file_size_mb(best),
        "onnx_fp32": file_size_mb(fp32_onnx),
        "onnx_fp16": file_size_mb(fp16_onnx),
        "onnx_int8": file_size_mb(int8_onnx),
    }

    result = {
        "weights": str(best),
        "data_yaml": str(data_yaml),
        "sizes_mb": sizes,
        "quantization_method": method,
        "accuracy_val": accuracy,
        "benchmark": bench,
        "notes": {
            "latency": "纯模型前向（640×640，batch=1，warmup 15 次后取平均）",
            "fp16": "FP16 需要 GPU / TensorRT / ORT-GPU 才能发挥速度优势",
            "int8": "INT8 静态量化使用验证集图片做 MinMax 校准",
        },
    }
    save_json(STAT_OUT / "quantization_benchmark.json", result)

    # 6) 图表
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    size_labels = list(sizes.keys())
    size_vals = [sizes[k] for k in size_labels]
    axes[0].bar(size_labels, size_vals, color="#4c72b0")
    axes[0].set_ylabel("模型大小 (MB)")
    axes[0].set_title("模型体积对比")
    axes[0].tick_params(axis="x", rotation=20)
    for i, v in enumerate(size_vals):
        axes[0].text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=9)

    bench_labels = [k for k in bench if bench[k].get("fps") is not None]
    fps_vals = [bench[k]["fps"] for k in bench_labels]
    axes[1].barh(bench_labels, fps_vals, color="#dd8452")
    axes[1].set_xlabel("FPS")
    axes[1].set_title("纯推理速度对比")
    for i, v in enumerate(fps_vals):
        axes[1].text(v + max(fps_vals) * 0.01, i, f"{v:.1f}", va="center",
                     fontsize=8)
    axes[1].tick_params(axis="y", labelsize=8)

    acc_labels, acc_vals = [], []
    for k, v in accuracy.items():
        if v:
            acc_labels.append(k)
            acc_vals.append(v["mAP50"])
    axes[2].bar(acc_labels, acc_vals, color="#55a868")
    axes[2].set_ylabel("mAP@0.5")
    axes[2].set_title("量化前后精度对比")
    axes[2].tick_params(axis="x", rotation=20)
    axes[2].set_ylim(0, 1.0)
    for i, v in enumerate(acc_vals):
        axes[2].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    fig.suptitle("YOLO 模型量化与导出对比（FP32 / FP16 / INT8）", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIG_OUT / "fig_task4_4_quantization.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    print(json.dumps({k: v for k, v in result.items() if k != "benchmark"},
                     ensure_ascii=False, indent=2))
    print(f"[完成] 量化结果: {STAT_OUT / 'quantization_benchmark.json'}")


if __name__ == "__main__":
    main()



