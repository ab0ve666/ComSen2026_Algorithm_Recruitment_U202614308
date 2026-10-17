# 算法组招新题目完成项目

本仓库完成了 2026 算法组招新题目中的：

| 任务 | 内容 | 状态 |
| --- | --- | --- |
| task-2 | OpenCV / Pillow 图像处理、轮廓检测、运动目标检测、小球追踪 | 完成 |
| task-3 | 深度学习概念、梯度下降最小二乘拟合、MLP 识别 MNIST | 完成 |
| task-4（题目一） | 基于 YOLO 的无人机目标检测、训练、视频推理、FP16/INT8 量化与 ONNX 导出 | 完成 |

> 说明：题目 PDF 中还包含 task-1（环境配置），本项目的运行环境即 task-1 的成果，详见
> [环境说明](#环境说明)。

## 项目结构

```text
cv-recruitment/
├─ data/                          # 题目素材（不纳入 git）
│  ├─ task2/                      # example1~4
│  ├─ task3/                      # MNIST parquet
│  ├─ task4/                      # drone.pt、example 图片/视频、yolo11n.pt
│  └─ drone_dataset/              # Roboflow 无人机数据集（YOLO 格式）
├─ task2_opencv_pillow/
│  ├─ scripts/
│  │  ├─ task2_image_ops.py       # 题1、题2
│  │  ├─ task2_contours.py        # 题3
│  │  ├─ task2_motion.py          # 题4
│  │  └─ task2_tracking.py        # 题5
│  └─ outputs/{images,videos,stats}
├─ task3_mlp_mnist/
│  ├─ src/
│  │  ├─ least_squares_fit.py     # 题2
│  │  └─ mnist_mlp.py             # 题3
│  └─ outputs/{figures,models,*.json,csv}
├─ task4_yolo_drone/
│  ├─ src/
│  │  ├─ 01_pretrained_inference.py
│  │  ├─ 02_train.py
│  │  ├─ 03_video_inference.py
│  │  ├─ 04_export_quantize.py
│  │  └─ common.py
│  └─ outputs/{models,predictions,videos,train,figures,stats}
├─ report/                        # 实验报告（docx / pdf）与图表
├─ requirements.txt
└─ README.md
```

## 环境说明（task-1）

- Python 3.13.15 + venv
- PyTorch 2.11.0 + CUDA 12.8（NVIDIA GeForce RTX 4060 Laptop 8GB）
- ultralytics 8.x、OpenCV（contrib）、NumPy / pandas / pyarrow / matplotlib /
  scikit-learn / python-docx / onnx / onnxruntime

```bash
python -m venv C:\Users\outof\venvs\cvhw
C:\Users\outof\venvs\cvhw\Scripts\activate
pip install -r requirements.txt
# 若 cv2 被 opencv-python 覆盖，补装 contrib 版本（提供 CSRT/KCF/MOSSE 跟踪器）：
pip install --force-reinstall opencv-contrib-python==4.12.0.88
```

运行环境验证：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import cv2; print(cv2.__version__, hasattr(cv2, 'TrackerCSRT_create'))"
```

## 数据准备

题目素材已放在 `data/` 下。无人机数据集来自题目给出的 Roboflow 链接：

```text
https://app.roboflow.com/ds/nSMQrzUp0X?key=TTcKC8vGXi
```

下载后将 zip 解压为 `data/drone_dataset/`（要求其中包含 `train/`、`valid/`、
`test/` 与 `data.yaml`）。官方 COCO 预训练权重 `yolo11n.pt` 放在
`data/task4/weights/`。

## 运行方法

```bash
# task-2
python task2_opencv_pillow/scripts/task2_image_ops.py
python task2_opencv_pillow/scripts/task2_contours.py
python task2_opencv_pillow/scripts/task2_motion.py
python task2_opencv_pillow/scripts/task2_tracking.py

# task-3
python task3_mlp_mnist/src/least_squares_fit.py
python task3_mlp_mnist/src/mnist_mlp.py --epochs 20

# task-4
python task4_yolo_drone/src/01_pretrained_inference.py
python task4_yolo_drone/src/02_train.py --epochs 40 --batch 16
python task4_yolo_drone/src/03_video_inference.py
python task4_yolo_drone/src/04_export_quantize.py
```

## 结果速览

完整图表与分析见 `report/算法组招新_实验报告.pdf`（DOCX 可编辑版同目录）。

### task-2

- 通道分离、几何变换：`task2_opencv_pillow/outputs/images/fig_task2_1_channels.png`、
  `fig_task2_2_transforms.png`。
- 飞机轮廓：面积 **71,326.5 px²**、周长 **2,197.7 px**（占图像 11.79%），
  凸包面积 114,174.5 px²；见 `outputs/stats/task2_contour_stats.json`。
- MOG2 vs 帧差法（example2，984 帧）：检测框填充率 **0.605 vs 0.484**，
  相邻帧掩膜 IoU **0.564 vs 0.335**；对比视频
  `outputs/videos/example2_comparison.mp4`。
- 小球跟踪（example3）：CSRT / KCF / MOSSE 成功率均 100%，平均 IoU
  0.910 / 0.902 / 0.906，速度 27.7 / 204.4 / 903.9 FPS；检测辅助混合跟踪
  IoU 0.930。
- example4（白色乒乓球、手部遮挡）：纯 CSRT 有效率 45.96%，模板+颜色+CSRT+
  Kalman 混合方法提升到 59.55%（50 次重初始化，5.7 FPS），报告详细分析了失败
  原因与改进方向。

### task-3

- 最小二乘拟合：**R² = 0.9516**、RMSE = 1.736；参数
  a1=2.013、a2=1.502、a3=2.940、a4=0.462、a5=−0.489、a6=0.760、b=2026.02
  （真值 2, 1.5, 3, 0.5, −0.5, 0.8, 2026）。
- MLP（784-512-256-10，537,354 参数）：测试准确率 **98.62%**，宏 F1 0.9861；
  20 epochs 在 RTX 4060 上约 33 秒。见 `outputs/task3_mnist_metrics.json`。

### task-4（题目一：YOLO 无人机检测）

- 官方 YOLO11n 预训练模型：bear 1、bus 8（bus/person/traffic light/car）、
  person 3（person/skis）；见 `outputs/predictions/pretrained_*.jpg`。
- 数据集：train 3893 / valid 649 / test 196；bird 与 drone 标注比例约 17:1；
  test 仅含 bird，无法评估 drone。
- 微调（drone.pt 起步，实际训练 20 轮，best epoch 16）：
  **val mAP@0.5 = 0.7307，mAP@0.5:0.95 = 0.3375，P = 0.7614，R = 0.7064**；
  val 匹配框分类准确率 0.9989；test（仅 bird）mAP@0.5 = 0.6092。
- 视频推理（378 帧）：初始 drone.pt 覆盖率 100% 但主要把目标判成 Bird
  （Drone 99 / Bird 279 帧）、框面积占比 48.5%；微调模型为 drone 311 / bird 23 帧，
  框面积占比 11.6%，置信度 0.6515→0.748，相邻帧 IoU 0.874→0.936，
  纯推理约 81→83 FPS，端到端 27.5→35.9 FPS。
- 量化与导出：ONNX FP32 10.11 MB / mAP 0.7201；FP16 5.11 MB / 0.7205；
  INT8（QDQ，排除输出 Concat）3.05 MB / 0.6576，CPU 延迟
  19.96→17.84 ms（约 1.12×）。详见
  `outputs/stats/quantization_benchmark.json`。

## 说明

- 所有脚本均可从项目根目录直接运行，路径基于 `Path(__file__)` 自动定位，不依赖当前
  工作目录。
- 服务器无图形界面，`cv2.imshow` 显示部分以保存的 PNG 代替。
- example4 的白色乒乓球无纹理、存在手部遮挡与镜头运动，传统跟踪器难以稳定跟踪；
  报告把它作为失败案例分析，并给出检测辅助、Kalman、深度跟踪器等改进方向。
- 数据集 test 划分只含 bird，故 test mAP 不能代表无人机检测性能；模型选择与
  分析以验证集（含 308 个 drone 标注）为准。
- 报告文档位于 `report/`，包含各题结果、指标表格与分析；`report/build_report.py`
  可从各 `outputs/stats/*.json` 一键重建报告。
