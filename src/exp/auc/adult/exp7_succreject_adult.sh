#!/bin/bash

# 定义 budget 列表
#  7 10 20  30  40  50  60  70  80  90  100 150 200 250  300  350  400  450  500  550  600  700  800  900  1000 1200 1400 1600 1800 2000
budgets=(
# 20 32 81  129 178 226 275 324 372 421
# 469 712 955 1198 1441
1684 1927 2170 2413 2656
2899 3385 3871 4357 4843 5815 6787 7759 8731 9703
)

# 顺序执行
for b in "${budgets[@]}"; do
    echo "🚀 正在运行: python ../exp7_only_2phase_succreject.py --budget=${b}"
    python ../exp7_only_2phase_succreject.py --dataset=adult --batch_size=64 --max_epochs=8 --budget=${b} --num_workers=4 --device_ids=1 --gpu_ids=1,1,1,1
    echo "✅ budget=${b} 执行完成，等待 3 秒..."
    sleep 3
done

echo "🎉 所有 budget 执行完毕！"
