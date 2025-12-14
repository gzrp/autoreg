import os
import re
import numpy as np
import matplotlib.pyplot as plt

def extract_metric_from_line(line, metric_type):
    """适配新日志格式的指标提取"""
    patterns = {
        "Train Loss": r"Train Loss:\s*([\d\.]+)",
        "Train AUC": r"Train Loss:\s*[\d\.]+,\s*Train Acc:\s*[\d\.]+,\s*Train AUC:\s*([\d\.]+)",

        "Val Loss": r"Val Loss:\s*([\d\.]+)",
        "Val AUC": r"Val Loss:\s*[\d\.]+,\s*Val Acc:\s*[\d\.]+,\s*Val AUC:\s*([\d\.]+)",
    }
    match = re.search(patterns[metric_type], line)
    return float(match.group(1)) if match else None


def plot_metrics(directory):
    metrics = ["Train Loss", "Train AUC", "Val Loss", "Val AUC"]
    files = sorted([f for f in os.listdir(directory) if f.endswith(".txt")])

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    axes = ax.flatten()

    for i, metric in enumerate(metrics):
        axes[i].set_title(metric)
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel(metric)
        axes[i].grid(True)

        for file in files:
            path = os.path.join(directory, file)

            values = []

            with open(path, "r") as f:
                for line in f:
                    v = extract_metric_from_line(line, metric)
                    if v is not None:
                        values.append(v)

            if len(values) == 0:
                continue

            epochs = list(range(1, len(values) + 1))
            label = os.path.splitext(file)[0]

            # 图例显示最后 epoch 数值
            axes[i].plot(epochs, values, label=f"{label} (last={values[-1]:.4f})", linewidth=1.0)

        axes[i].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(directory, "summary_metrics.png"), dpi=300)
    plt.show()


if __name__ == "__main__":
    log_dir = "/data/ruipeng/workdir/autoreg/script/auc/adult"
    plot_metrics(log_dir)
