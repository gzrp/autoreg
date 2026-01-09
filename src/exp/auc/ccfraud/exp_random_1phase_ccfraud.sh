#!/bin/bash

# 定义 budget 列表
budgets=(
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 25 30 35 40 45 50 55 60 65 67
70 80 90 100 120 140 150 160 180
210 240 270 300 330 360 420 480 540 600 720 900 1080
1200 1500 1800 2100 2400 2700 3000
3600 4200 4800 5400 6000 6600 7200
#8400 9600 10800 12000 13200 14219
)

# 顺序执行
for b in "${budgets[@]}"; do
    echo "🚀 正在运行: python exp_random_1phase.py --budget=${b}"
    python ../exp_random_1phase.py --dataset=ccfraud --batch_size=256 --max_epochs=16 --grace_period=1 --sample_ratio=0.20001 --num_workers=4 --budget=${b} --device_ids=0,1 --gpu_ids=0,1,0,1
    echo "✅ budget=${b} 执行完成，等待 3 秒..."
    sleep 3
done

echo "🎉 所有 budget 执行完毕！"
