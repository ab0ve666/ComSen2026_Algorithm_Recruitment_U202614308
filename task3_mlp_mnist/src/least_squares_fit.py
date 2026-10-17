"""task-3 题 2：使用 PyTorch 自动求导 + 梯度下降完成二元函数的最小二乘拟合。

待拟合函数（题目给定）：
    y = 2*x1^2 + 1.5*x1*x2 + 3*sin(x1) + 0.5*cos(2*x2)
        + exp(-0.5*x1) + 0.8*x2 + 2026 + noise,   noise ~ U(-3, 3)

拟合形式：
    y_hat = a1*x1^2 + a2*x1*x2 + a3*sin(x1) + a4*cos(2*x2)
            + exp(a5*x1) + a6*x2 + b

优化：Adam（自动求导）+ LBFGS 精调；输出参数、R²、RMSE，并绘制真实/拟合三维曲面。

运行：
    python task3_mlp_mnist/src/least_squares_fit.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "task3_mlp_mnist" / "outputs"
FIG = OUT / "figures"
STAT = OUT

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
torch.set_default_dtype(torch.float64)

TRUE = {"a1": 2.0, "a2": 1.5, "a3": 3.0, "a4": 0.5,
        "a5": -0.5, "a6": 0.8, "b": 2026.0}


def true_function(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return (TRUE["a1"] * x1 ** 2 + TRUE["a2"] * x1 * x2
            + TRUE["a3"] * torch.sin(x1) + TRUE["a4"] * torch.cos(2 * x2)
            + torch.exp(TRUE["a5"] * x1) + TRUE["a6"] * x2 + TRUE["b"])


def model(params: dict[str, torch.Tensor], x1: torch.Tensor,
          x2: torch.Tensor) -> torch.Tensor:
    exponent = torch.clamp(params["a5"] * x1, -20.0, 20.0)
    return (params["a1"] * x1 ** 2 + params["a2"] * x1 * x2
            + params["a3"] * torch.sin(x1) + params["a4"] * torch.cos(2 * x2)
            + torch.exp(exponent) + params["a6"] * x2 + params["b"])


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(2026)
    rng = np.random.default_rng(2026)

    n = 4000
    x_np = rng.uniform(-3.0, 3.0, size=(n, 2))
    x1 = torch.tensor(x_np[:, 0])
    x2 = torch.tensor(x_np[:, 1])
    noise = torch.tensor(rng.uniform(-3.0, 3.0, size=n))
    y_true = true_function(x1, x2) + noise
    print(f"[数据] {n} 个采样点，x1/x2∈[-3,3]，噪声 U(-3,3)，"
          f"y 范围 [{y_true.min():.1f}, {y_true.max():.1f}]")

    init = {"a1": 1.0, "a2": 1.0, "a3": 1.0, "a4": 1.0,
            "a5": -0.2, "a6": 1.0, "b": 0.0}
    params = {k: torch.tensor(v, requires_grad=True) for k, v in init.items()}

    # ---------- 阶段 1：Adam ----------
    optimizer = torch.optim.Adam(params.values(), lr=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4000)
    loss_history = []
    t0 = time.perf_counter()
    for step in range(4001):
        optimizer.zero_grad()
        pred = model(params, x1, x2)
        loss = torch.mean((pred - y_true) ** 2)
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_history.append(float(loss.detach()))
        if step % 500 == 0:
            print(f"  Adam step {step:5d}  MSE={loss.item():.4f}")
    print(f"[阶段1] Adam 完成，耗时 {time.perf_counter() - t0:.1f}s，"
          f"MSE={loss_history[-1]:.4f}")

    # ---------- 阶段 2：LBFGS 精调 ----------
    optimizer2 = torch.optim.LBFGS(params.values(), lr=0.5, max_iter=200,
                                   history_size=50,
                                   line_search_fn="strong_wolfe")

    def closure():
        optimizer2.zero_grad()
        pred = model(params, x1, x2)
        loss = torch.mean((pred - y_true) ** 2)
        loss.backward()
        return loss

    t1 = time.perf_counter()
    final_loss = optimizer2.step(closure)
    print(f"[阶段2] LBFGS 完成，耗时 {time.perf_counter() - t1:.1f}s，"
          f"MSE={final_loss.item():.6f}")

    with torch.no_grad():
        y_pred = model(params, x1, x2).numpy()
    y_true_np = y_true.numpy()
    r2 = r2_score(y_true_np, y_pred)
    rmse = float(np.sqrt(np.mean((y_true_np - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true_np - y_pred)))

    fitted = {k: float(v.detach()) for k, v in params.items()}
    table = []
    for k in ["a1", "a2", "a3", "a4", "a5", "a6", "b"]:
        table.append({
            "param": k,
            "true": TRUE[k],
            "fitted": round(fitted[k], 6),
            "abs_error": round(abs(fitted[k] - TRUE[k]), 6),
            "rel_error_percent": round(abs(fitted[k] - TRUE[k])
                                       / (abs(TRUE[k]) + 1e-12) * 100, 4),
        })

    result = {
        "n_samples": n,
        "noise": "U(-3, 3)",
        "true_params": TRUE,
        "fitted_params": {k: round(v, 6) for k, v in fitted.items()},
        "param_table": table,
        "R2": round(r2, 6),
        "RMSE": round(rmse, 6),
        "MAE": round(mae, 6),
        "final_MSE": round(float(np.mean((y_true_np - y_pred) ** 2)), 6),
        "adam_steps": 4000,
        "lbfgs_max_iter": 200,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
    }
    with open(STAT / "task3_least_squares.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # ---------- 绘图 ----------
    g = np.linspace(-3, 3, 120)
    X1, X2 = np.meshgrid(g, g)
    t1g = torch.tensor(X1.ravel())
    t2g = torch.tensor(X2.ravel())
    with torch.no_grad():
        Z_true = true_function(t1g, t2g).numpy().reshape(X1.shape) - 2026.0
        Z_fit = model(params, t1g, t2g).numpy().reshape(X1.shape) - 2026.0

    fig = plt.figure(figsize=(12, 11))
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1.plot_surface(X1, X2, Z_true, cmap="viridis", linewidth=0, antialiased=True,
                     alpha=0.95)
    idx = rng.choice(n, 250, replace=False)
    ax1.scatter(x_np[idx, 0], x_np[idx, 1], (y_true_np[idx] - 2026.0),
                c="red", s=8, alpha=0.6, label="含噪声采样点")
    ax1.set_title(f"真实函数（不含噪声）  R^2=1.0000\n"
                  f"z = y - 2026，便于观察", fontsize=11)
    ax1.set_xlabel("$x_1$")
    ax1.set_ylabel("$x_2$")
    ax1.set_zlabel("$z$")
    ax1.view_init(elev=28, azim=-120)

    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    ax2.plot_surface(X1, X2, Z_fit, cmap="plasma", linewidth=0, antialiased=True,
                     alpha=0.95)
    ax2.set_title(f"PyTorch 拟合函数  R^2={r2:.4f}\n"
                  f"RMSE={rmse:.3f}", fontsize=11)
    ax2.set_xlabel("$x_1$")
    ax2.set_ylabel("$x_2$")
    ax2.set_zlabel("$z$")
    ax2.view_init(elev=28, azim=-120)

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(loss_history, color="#1f77b4", lw=1.2)
    ax3.set_yscale("log")
    ax3.set_xlabel("Adam 迭代步数")
    ax3.set_ylabel("MSE（对数坐标）")
    ax3.set_title("优化过程损失曲线")
    ax3.grid(alpha=0.3)
    text = "\n".join([
        "拟合参数：",
        f"a1={fitted['a1']:.3f}  (真值 2)",
        f"a2={fitted['a2']:.3f}  (真值 1.5)",
        f"a3={fitted['a3']:.3f}  (真值 3)",
        f"a4={fitted['a4']:.3f}  (真值 0.5)",
        f"a5={fitted['a5']:.3f}  (真值 -0.5)",
        f"a6={fitted['a6']:.3f}  (真值 0.8)",
        f"b ={fitted['b']:.2f}  (真值 2026)",
        f"R^2 = {r2:.4f}",
    ])
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")
    ax4.text(0.02, 0.98, text, transform=ax4.transAxes, va="top", fontsize=10.5,
             bbox=dict(boxstyle="round", facecolor="#F7F7F7", alpha=0.95))

    fig.suptitle("task-3 题 2：梯度下降最小二乘拟合（真实曲面 vs 拟合曲面）", fontsize=15, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG / "fig_task3_2_least_squares.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[完成] 图: {FIG / 'fig_task3_2_least_squares.png'}")


if __name__ == "__main__":
    main()



