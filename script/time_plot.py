
import json
import matplotlib.pyplot as plt
import glob
import os

def load_curve(path):
    xs, ys = [], []
    with open(path, 'r') as f:
        for line in f:
            obj = json.loads(line)
            elapsed_min = obj["elapsed_time"] / 60.0   # 秒 -> 分钟
            metric = obj["best_metric"]
            xs.append(elapsed_min)
            ys.append(metric)
    return xs, ys


if __name__ == '__main__':

    # 读取目录下所有 jsonl 文件
    # /data/ruipeng/workdir/autoreg/exp/exp2/ccfraud
    files = glob.glob("/data/ruipeng/workdir/autoreg/exp/exp2/ccfraud/*.jsonl")

    plt.figure(figsize=(8, 5))

    for path in files:
        xs, ys = load_curve(path)
        name = os.path.basename(path)
        plt.plot(xs, ys, label=name, linewidth=2)

    # 对数坐标轴
    plt.xscale("log")

    # 指定刻度
    plt.xticks([1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
               [r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$",
                r"$10^{1}$", r"$10^{2}$", r"$10^{3}$"])

    plt.xlabel("Elapsed Time (min, log scale)")
    plt.ylabel("Best Metric")
    plt.title("Training Curves")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
