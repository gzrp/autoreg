#!/bin/bash

# 定义 budget 列表
budgets=(
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 22 24 25 26 28 30 32 33
35 40 45 50 55 60 70 80 90 100 110 120 130 140 150 160 170 180
210 240 270 300 330 360 420 480 540 600 720 840 960 1080 1200
1500 1800 2100 2400 2700 3000 3600 4200 4500 4800
5400 6000 6600 7200 7852
)

# 顺序执行
for b in "${budgets[@]}"; do
    echo "🚀 正在运行: python exp2_1phase.py --budget=${b}"
    python exp2_1phase.py --dataset=bank --batch_size=64 --max_epochs=16 --grace_period=1 --budget=${b} --device_ids=0,1
    echo "✅ budget=${b} 执行完成，等待 5 秒..."
    sleep 5
done

echo "🎉 所有 budget 执行完毕！"
