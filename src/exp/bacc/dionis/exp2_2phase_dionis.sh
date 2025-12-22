#!/bin/bash

# 定义 budget 列表
#1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 25 30 35 40 45 50 55 60
#70 80 90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250
budgets=(
252 270 300 330 360 420 480 540 600 660 720 900 1080
1200 1500 1800 2100 2400 2700 3000
3600 4200 4800 5400 6000 6600 7200 8400 9600 10800 12000 13200
15000 16800 18600 20400 22200 24000 25800 27741
)

# 顺序执行
for b in "${budgets[@]}"; do
    echo "🚀 正在运行: python exp2_2phase.py --budget=${b}"
    python ../exp2_2phase.py --dataset=dionis --batch_size=256 --max_epochs=32 --grace_period=1 --budget=${b} --device_ids=1,1 --gpu_ids=1,1,1,1 --storage="~/autodl-tmp/ray_results"
    echo "✅ budget=${b} 执行完成，等待 3 秒..."
    sleep 3
done

echo "🎉 所有 budget 执行完毕！"
