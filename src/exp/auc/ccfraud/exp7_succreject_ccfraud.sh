#!/bin/bash

# 定义 budget 列表
# 11  20  30  40  50  60  70   80   90
# 100  150  200   250  300  350  400  450  500  550
# 600  700   800   900   1000   1200   1400   1600   1800   2000
# succrejt     66 205  368 532 695 858 1022 1185 1349  1512 2329 3146 3963  4780 5597 6414 7231 8048 8865 9682 11316 12950 14584 16218 17852 19486 21119 22753 24387 26021 27655 29289 30923 32557

budgets=(
66 205  368 532 695 858 1022 1185 1349
1512 2329 3146 3963  4780 5597 6414 7231 8048 8865
9682 11316 12950 14584 16218 19486 22753 26021 29289 32557
)

# 顺序执行
for b in "${budgets[@]}"; do
    echo "🚀 正在运行: python ../exp7_only_2phase_succreject.py --budget=${b}"
    python ../exp7_only_2phase_succreject.py --dataset=ccfraud --batch_size=256 --max_epochs=16 --budget=${b} --num_workers=4 --device_ids=0,1 --gpu_ids=0,1,0,1
    echo "✅ budget=${b} 执行完成，等待 3 秒..."
    sleep 3
done

echo "🎉 所有 budget 执行完毕！"
