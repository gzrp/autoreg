#!/bin/bash

# 定义 budget 列表
budgets=(
1 2 3 4 5 6 7 8 9 10 12 14 16 18 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100
110 120 130 140 150 160 170 177 180 190 200 225 250 275 300 350 400 450 500
600 700 800 900 1000 1200 1400 1600 1800 2000 2500 3000 3500 4000 4500 5000 5500 6000 6654
)

# 顺序执行
for b in "${budgets[@]}"; do
    echo "🚀 正在运行: python reg_selection_time_only_one_phase_new.py --budget=${b}"
    python reg_selection_time_only_one_phase_new.py --dataset=frappe --batch_size=128 --max_epochs=16 --budget=${b}
    echo "✅ budget=${b} 执行完成，等待 5 秒..."
    sleep 5
done

echo "🎉 所有 budget 执行完毕！"
