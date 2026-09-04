import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def smooth(y, weight=0.9):
    y = list(y)
    if not y:
        return y
    smoothed = [y[0]]
    for i in range(1, len(y)):
        smoothed.append(smoothed[-1] * weight + y[i] * (1 - weight))
    return smoothed

def plot_results(csv_path, save_path=None):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # 横轴
    if "epoch" in df.columns:
        x = df["epoch"]
    else:
        x = range(len(df))

    # 按 Ultralytics 常见列名来画
    cols = [
        "train/box_loss",
        "train/cls_loss",
        "train/dfl_loss",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "val/box_loss",
        "val/cls_loss",
        "val/dfl_loss",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]

    cols = [c for c in cols if c in df.columns]

    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    axes = axes.ravel()

    for i, col in enumerate(cols):
        y = pd.to_numeric(df[col], errors="coerce")

        axes[i].plot(x, y, "o-", label="results", markersize=4, linewidth=2)
        axes[i].plot(x, smooth(y.fillna(method="ffill").fillna(method="bfill")), ":", label="smooth", linewidth=2.5)

        axes[i].set_title(col)
        axes[i].grid(True)

        if i == 1:
            axes[i].legend()

    # 多出来的空子图关掉
    for j in range(len(cols), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    if save_path is None:
        save_path = csv_path.parent / "my_results.png"

    plt.savefig(save_path, dpi=200)
    print(f"saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    plot_results("/home/guiyan/ztz/repo/ultralytics-main/runs/detect/DroneVehicle_train100/results.csv")