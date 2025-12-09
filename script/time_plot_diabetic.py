import json
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter
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
    plt.figure(figsize=(6, 5))
    dataset = "diabetic"
    Dataset = "Diabetic"
    legend_map = {
        "2Phase": f"/data/ruipeng/workdir/autoreg/exp/exp2/{dataset}/2phase_time_log_extract.jsonl",
        "ASHA": f"/data/ruipeng/workdir/autoreg/exp/exp2/{dataset}/asha_time_log_extract.jsonl",
        "BOHB": f"/data/ruipeng/workdir/autoreg/exp/exp2/{dataset}/bohb_time_log_extract.jsonl",
        "Hyperband": f"/data/ruipeng/workdir/autoreg/exp/exp2/{dataset}/hyperband_time_log_extract.jsonl",
        "1Phase-AgE": f"/data/ruipeng/workdir/autoreg/exp/exp2/{dataset}/1phase_time_log_extract.jsonl",
    }

    color_map = {
        "2Phase": "forestgreen",
        "ASHA": "darkorchid",
        "BOHB": "royalblue",
        "Hyperband": "darkcyan",
        "1Phase-AgE": "darkorange"
    }

    markers = ['o', 'X', 's', 'h', '>', 'D', '*', 'X', 'P', '^', '>']

    for i, (legend_name, file_path) in enumerate(legend_map.items()):
        if not os.path.exists(file_path):
            print(f"⚠️ 文件不存在: {file_path}")
            continue

        xs, ys = load_curve(file_path)
        color = color_map.get(legend_name, None)

        plt.plot(
            xs, ys,
            label=legend_name,
            linewidth=2,
            color=color,
            marker=markers[i % len(markers)],
            markersize=8,
            markerfacecolor='auto',
            markeredgewidth=0.8
        )

    # 横轴对数坐标
    plt.xscale("log")

    # 横轴主刻度标签
    plt.xticks(
        [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
        [r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$",
         r"$10^{1}$", r"$10^{2}$", r"$10^{3}$"]
    )

    ax = plt.gca()

    # 横轴主次刻度
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=range(2, 10)))

    # 网格线
    ax.grid(True, which='major', axis='both', linestyle='-', alpha=0.6)
    ax.grid(True, which='minor', axis='both', linestyle='--', alpha=0.4)

    # baseline 水平线 0.933067
    mlp_plain = 0.608794
    ax.axhline(y=mlp_plain, color='red', linestyle='-', linewidth=1.5, label="MLP-Plain")

    # =============================
    # ⭐⭐⭐ 关键部分：Y 轴 ×100 并保留两位小数
    # =============================
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y * 100:.2f}"))
    # =============================

    plt.xlabel("Response Time Threshold $T_{max}$ (min)")
    plt.ylabel(f"Balanced ACC (%) on {Dataset}")   # 你可写成 (%)，可选
    plt.title("SLO-aware of 2Phase-RS")

    # 纵轴刻度
    ax.tick_params(axis='y', which='major', direction='inout', length=6, width=1)
    ax.tick_params(axis='y', which='minor', direction='inout', length=4, width=0.8)

    plt.legend(fontsize=9, title_fontsize=10)
    plt.tight_layout()

    # 保存图片
    save_path = f"/data/ruipeng/workdir/autoreg/exp/exp2/{dataset}/{dataset}_slo_2phase_rs.png"
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    print(f"✅ 图像已保存到: {save_path}")
    plt.show()
