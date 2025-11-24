#!/bin/bash

# 定义 budget 列表
budgets=(
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
17 20 30 40 50 60 70 80 90
100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950
1000 1100 1200 1300 1400 1500 1600 1664
)

# 顺序执行
for b in "${budgets[@]}"; do
    echo "🚀 正在运行: python reg_selection_time_only_one_phase.py --budget=${b}"
    python reg_selection_time_only_one_phase.py --dataset=connect --budget=${b}
    echo "✅ budget=${b} 执行完成，等待 5 秒..."
    sleep 5
done

echo "🎉 所有 budget 执行完毕！"
