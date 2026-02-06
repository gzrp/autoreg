#!/bin/bash

# 定义 budget 列表
# 4  10  20   30   40   50   60   70   80    90    100   150  200   250   300   350   400   450   500   550   600   700   800   900   1000  1200  1400  1600  1800    2000
#237 593 1185 1777 2369 2961 3553 4145 4737 5329  5922  8882 11843 14803 17764 20724 23685 26645 29606 32567 35527 41448 47369 53290 59211 71054 82896 94738 106580 118422

budgets=(
118422
)

# 顺序执行
for b in "${budgets[@]}"; do
    echo "🚀 正在运行: python ../exp7_only_2phase_all.py --budget=${b}"
    python ../exp7_only_2phase_all.py --dataset=frappe --batch_size=256 --max_epochs=16 --budget=${b} --num_workers=4 --device_ids=1,2,3 --gpu_ids=1,2,3,0,1,2,3
    echo "✅ budget=${b} 执行完成，等待 3 秒..."
    sleep 3
done

echo "🎉 所有 budget 执行完毕！"
