import matplotlib.pyplot as plt

if __name__ == '__main__':

    # 表格数据
    data = [
        ["MLP", "0.637087", "-", "-"],
        ["ASHA", "0.642134", "11240.19 s", "5.16×"],
        ["HyperBand", "0.642345", "30467.43 s", "13.97×"],
        ["BOHB", "0.642475", "26011.46 s", "11.93×"],
        ["AgE-ASHA(ours)", "0.642279", "2180.40 s", "1.00×"],
    ]

    columns = ["Baseline", "Balanced Acc", "Time usage", "Speed up"]

    # 创建图形
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis('off')  # 不显示坐标轴

    # 绘制表格
    table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center')

    # 美化表格
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)

    # 设置列宽
    for i in range(len(columns)):
        table.auto_set_column_width(col=list(range(len(columns))))

    # 设置表头加粗
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#f2f2f2')

    # 添加标题
    plt.title("Clickpred Dataset", fontsize=13, weight='bold', pad=1)

    plt.tight_layout()
    plt.savefig("/data/ruipeng/workdir/autoreg/.raw/exp2/clickpred/clickpred_results_table.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ 表格已保存为：clickpred_results_table.png")
