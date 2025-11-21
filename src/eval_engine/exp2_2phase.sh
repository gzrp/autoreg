#!/bin/bash

# 定义 budget 列表
budgets=(
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
21 22 23 24 25 26 27 28
30 40 50 60 70 80 90
100 200 300 400 500 600 700 800 900 1000
1200 1400 1600 1800 2000 2200 2400 2600 2650
)

# 顺序执行
for b in "${budgets[@]}"; do
    echo "🚀 正在运行: python reg_selection_time.py --budget=${b}"
    python reg_selection_time.py --budget=${b}
    echo "✅ budget=${b} 执行完成，等待 5 秒..."
    sleep 5
done

echo "🎉 所有 budget 执行完毕！"
