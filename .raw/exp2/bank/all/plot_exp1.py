import matplotlib.pyplot as plt

if __name__ == '__main__':

    # 表格数据
    data = [
        ["MLP", "0.882752", "-", "-"],
        ["ASHA", "0.892100", "4643.79 s", "2.85x"],
        ["HyperBand", "0.892286", "8343.06 s", "5.13×"],
        ["BOHB", "0.892357", "11159.56 s", "6.86×"],
        ["AgE-ASHA(ours)", "0.892585", "1626.98 s", "1.00×"],
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
    plt.title("Bank Dataset", fontsize=13, weight='bold', pad=1)

    plt.tight_layout()
    plt.savefig("/data/ruipeng/workdir/autoreg/.raw/exp2/bank/bank_results_table.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ 表格已保存为：bank_results_table.png")
