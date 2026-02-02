#!/bin/bash

# 定义 budget 列表
# uniform
# 4   10  20   30   40   50   60   70   80    90    100   150  200   250   300   350   400   450   500   550   600   700   800   900   1000  1200  1400  1600  1800    2000
# 249 623 1245 1868 2490 3112 3735 4357 4979 5602  6224  9336 12448 15560 18672 21783 24895 28007 31119 34231 37343 43566 49790 56014 62237 74685 87132 99579  112027 124474

budgets=(
124474
)

# 顺序执行
for b in "${budgets[@]}"; do
    echo "🚀 正在运行: python ../exp7_only_2phase_all.py --budget=${b}"
    python ../exp7_only_2phase_all.py --dataset=dionis --batch_size=256 --max_epochs=32 --budget=${b} --num_workers=4 --device_ids=3 --gpu_ids=3
    echo "✅ budget=${b} 执行完成，等待 3 秒..."
    sleep 3
done

echo "🎉 所有 budget 执行完毕！"
q