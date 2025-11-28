#!/bin/bash

# 定义 budget 列表
#budgets=(
#1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
#31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52
#60 70 80 90 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950
#1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 2100 2181
#)

budgets=(
1300 1400 1500 1600 1700 1800 1900 2000 2100 2181
)


# 顺序执行
for b in "${budgets[@]}"; do
    echo "🚀 正在运行: python reg_selection_time.py --budget=${b}"
    python reg_selection_time.py --dataset=clickpred --batch_size=256 --num_samples=2000 --max_epochs=16 --grace_period=1 --budget=${b}
    echo "✅ budget=${b} 执行完成，等待 5 秒..."
    sleep 5
done

echo "🎉 所有 budget 执行完毕！"
