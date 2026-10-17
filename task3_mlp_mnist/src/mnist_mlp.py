"""task-3 题 3：PyTorch + pandas 构建 MLP 识别 MNIST 手写数字。

数据：data/task3/train-00000-of-00001.parquet（60000 张 28×28 PNG，字段 image/label）
划分：按 8:1:1 分层划分为训练 / 验证 / 测试集。
模型：784 → 512 → 256 → 10 的全连接 MLP（BatchNorm + Dropout）。
输出：Loss / Train Acc / Test Acc 曲线、混淆矩阵、分类指标、≥3 张预测可视化、模型权重。

运行：
    python task3_mlp_mnist/src/mnist_mlp.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "task3" / "train-00000-of-00001.parquet"
OUT = ROOT / "task3_mlp_mnist" / "outputs"
FIG = OUT / "figures"
MODEL_DIR = OUT / "models"

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_mnist(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """用 pandas 读取 parquet，用 OpenCV 解码其中的 PNG 字节。"""
    df = pd.read_parquet(path)
    print(f"[数据] parquet shape={df.shape}, columns={df.columns.tolist()}")
    images = np.empty((len(df), 28, 28), dtype=np.uint8)
    for i, item in enumerate(df["image"].to_numpy()):
        buf = np.frombuffer(item["bytes"], dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"第 {i} 张图片解码失败")
        images[i] = img
    labels = df["label"].to_numpy(dtype=np.int64)
    print(f"[数据] 解码完成 images={images.shape}, labels={labels.shape}, "
          f"类别分布={np.bincount(labels).tolist()}")
    return images, labels


def to_tensor(images: np.ndarray, labels: np.ndarray) -> TensorDataset:
    x = torch.from_numpy(images.astype(np.float32) / 255.0).unsqueeze(1)
    x = (x - 0.1307) / 0.3081  # MNIST 全局均值/标准差
    y = torch.from_numpy(labels.astype(np.int64))
    return TensorDataset(x, y)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             criterion: nn.Module) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_pred, all_true = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * yb.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
        all_pred.append(pred.cpu().numpy())
        all_true.append(yb.cpu().numpy())
    return (total_loss / total, correct / total,
            np.concatenate(all_pred), np.concatenate(all_true))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    for d in (FIG, MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] {device} {torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}")

    images, labels = load_mnist(DATA)
    x_tmp, x_test, y_tmp, y_test = train_test_split(
        images, labels, test_size=0.10, stratify=labels, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(
        x_tmp, y_tmp, test_size=1 / 9, stratify=y_tmp, random_state=42)
    print(f"[划分] train={len(x_train)}, val={len(x_val)}, test={len(x_test)}")

    train_ds = to_tensor(x_train, y_train)
    val_ds = to_tensor(x_val, y_val)
    test_ds = to_tensor(x_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0)
    train_eval_loader = DataLoader(train_ds, batch_size=512, shuffle=False, num_workers=0)

    model = MLP().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[模型] MLP 参数量={n_params:,}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [],
               "val_acc": [], "test_loss": [], "test_acc": [], "lr": []}
    best_val_acc = 0.0
    best_state = None
    best_epoch = 0
    t0 = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * yb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)
        scheduler.step()

        train_loss = running_loss / total
        train_acc = correct / total
        _, train_acc_full, _, _ = evaluate(model, train_eval_loader, device, criterion)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, device, criterion)
        test_loss, test_acc, _, _ = evaluate(model, test_loader, device, criterion)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc_full)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch:2d}/{args.epochs}  loss={train_loss:.4f}  "
              f"train={train_acc_full:.4f}  val={val_acc:.4f}  test={test_acc:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

    elapsed = time.perf_counter() - t0
    assert best_state is not None
    model.load_state_dict(best_state)
    torch.save(best_state, MODEL_DIR / "mnist_mlp.pth")
    pd.DataFrame(history).to_csv(OUT / "task3_mnist_history.csv", index=False)

    test_loss, test_acc, y_pred, y_true = evaluate(model, test_loader, device, criterion)
    train_loss, train_acc, _, _ = evaluate(model, train_eval_loader, device, criterion)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    report = classification_report(y_true, y_pred, digits=4, output_dict=True,
                                   zero_division=0)
    per_class = [{
        "digit": d,
        "precision": round(report[str(d)]["precision"], 4),
        "recall": round(report[str(d)]["recall"], 4),
        "f1": round(report[str(d)]["f1-score"], 4),
        "support": int(report[str(d)]["support"]),
    } for d in range(10)]

    result = {
        "dataset": "MNIST (parquet, 60000)",
        "split": {"train": len(x_train), "val": len(x_val), "test": len(x_test)},
        "model": "MLP 784-512-256-10 (BatchNorm + Dropout)",
        "params": int(n_params),
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "best_epoch": best_epoch,
        "best_val_acc": round(best_val_acc, 4),
        "train_acc": round(train_acc, 4),
        "test_acc": round(test_acc, 4),
        "test_loss": round(test_loss, 4),
        "train_loss": round(train_loss, 4),
        "elapsed_seconds": round(elapsed, 2),
        "per_class": per_class,
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "confusion_matrix": cm.tolist(),
    }
    with open(OUT / "task3_mnist_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # ---------- Loss / Acc 曲线 ----------
    ep = history["epoch"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 10.5))
    axes[0].plot(ep, history["train_loss"], "o-", label="训练损失", ms=3)
    axes[0].plot(ep, history["val_loss"], "s-", label="验证损失", ms=3)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("交叉熵损失")
    axes[0].set_title("Loss 曲线")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(ep, history["train_acc"], "o-", label="Train Acc", ms=3)
    axes[1].plot(ep, history["val_acc"], "s-", label="Val Acc", ms=3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("准确率")
    axes[1].set_title("训练 / 验证准确率")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(ep, history["test_acc"], "^-", color="#d62728", label="Test Acc", ms=4)
    axes[2].axhline(test_acc, ls="--", color="gray", lw=1,
                    label=f"最终测试准确率 {test_acc:.4f}")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("准确率")
    axes[2].set_title("测试准确率")
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    fig.suptitle("task-3 题 3：MLP 训练过程", fontsize=15, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG / "fig_task3_3_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------- 混淆矩阵 ----------
    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(f"MNIST 测试集混淆矩阵（准确率 {test_acc:.4f}）", fontsize=13)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=7.5,
                    color="white" if cm[i, j] > cm.max() * 0.45 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIG / "fig_task3_3_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------- 每类指标 ----------
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(10)
    width = 0.27
    ax.bar(x - width, [p["precision"] for p in per_class], width, label="Precision")
    ax.bar(x, [p["recall"] for p in per_class], width, label="Recall")
    ax.bar(x + width, [p["f1"] for p in per_class], width, label="F1-score")
    ax.set_xticks(x)
    ax.set_ylim(0.9, 1.005)
    ax.set_xlabel("数字类别")
    ax.set_ylabel("指标值")
    ax.set_title("各类别 Precision / Recall / F1")
    ax.legend(ncol=3, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_task3_3_per_class.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------- 预测可视化（8 张，包含错误样本） ----------
    model.eval()
    rng = np.random.default_rng(7)
    correct_pool = np.where(y_pred == y_true)[0]
    wrong_pool = np.where(y_pred != y_true)[0]
    n_wrong = min(3, len(wrong_pool))
    n_correct = 8 - n_wrong
    indices = np.concatenate([
        rng.choice(wrong_pool, n_wrong, replace=False) if n_wrong else np.array([], int),
        rng.choice(correct_pool, n_correct, replace=False),
    ]).astype(int)
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.4))
    with torch.no_grad():
        for ax, idx in zip(axes.ravel(), indices):
            img = x_test[idx]
            tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
            tensor = ((tensor - 0.1307) / 0.3081).unsqueeze(1).to(device)
            prob = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
            pred = int(prob.argmax())
            conf = float(prob[pred])
            ax.imshow(img, cmap="gray")
            ok = pred == y_test[idx]
            ax.set_title(f"真值 {y_test[idx]} / 预测 {pred}\n置信度 {conf:.3f}",
                         color="green" if ok else "red", fontsize=10)
            ax.axis("off")
    fig.suptitle(f"MNIST 手写数字识别结果可视化（{n_wrong} 个错误样本 + "
                 f"{n_correct} 个正确样本）", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG / "fig_task3_3_predictions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps({k: v for k, v in result.items() if k != "confusion_matrix"},
                     ensure_ascii=False, indent=2))
    print(f"[完成] 模型: {MODEL_DIR / 'mnist_mlp.pth'}")
    print(f"[完成] 图: {FIG}")


if __name__ == "__main__":
    main()

