#!/bin/bash

# 定义 budget 列表
budgets=(
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 18 20 22 24 26 28 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100
110 120 130 140 150 160 170 180 190 200 220 240 260 280 300 320 340 360 380 400 420 440 460 480
500 550 600 650 700 750 800 850 900 950 1000 1050 1100 1150 1200 1250 1300 1350 1400 1450 1500 1550 1600 1627
)

# 顺序执行
for b in "${budgets[@]}"; do
    echo "🚀 正在运行: python reg_selection_time.py --budget=${b}"
    python reg_selection_time.py --dataset=bank --batch_size=64 --max_epochs=4 --grace_period=2 --budget=${b}
    echo "✅ budget=${b} 执行完成，等待 5 秒..."
    sleep 5
done

echo "🎉 所有 budget 执行完毕！"
