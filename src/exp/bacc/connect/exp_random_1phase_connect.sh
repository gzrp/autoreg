#!/bin/bash

# 定义 budget 列表
budgets=(
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 23 25 27 30 32 35 37 40 42 45 47 50 53
54 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240
270 300 330 360 390 420 450 480 510 540 600 660 720 780 840 900 960 1020 1080
1200 1350 1500 1650 1800 1950 2100 2250 2400 2550 2700 1850 3000
3300 3600 3900 4200 4500 4800 5100 5400 5700 6000 6300 6600 6900 7200
#7800 8400 9000 9600 10200 10800 11912
)

# 顺序执行
for b in "${budgets[@]}"; do
    echo "🚀 正在运行: python exp_random_1phase.py --budget=${b}"
    python ../exp_random_1phase.py --dataset=connect --batch_size=64 --max_epochs=16 --grace_period=1 --num_workers=4 --budget=${b} --device_ids=0,1 --gpu_ids=0,1,0,1
    echo "✅ budget=${b} 执行完成，等待 3 秒..."
    sleep 3
done

echo "🎉 所有 budget 执行完毕！"
