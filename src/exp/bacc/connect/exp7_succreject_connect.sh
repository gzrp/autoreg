#!/bin/bash

# 定义 budget 列表
# 11  20  30  40  50  60  70   80   90
# 100  150  200   250  300  350  400  450  500  550
# 600  700   800   900   1000   1200   1400   1600   1800   2000
# succrejct       53 163 293 423 553 683 813  943  1073   1204 1854 2504 3154  3805 4455 5105 5756 6406 7056 7707 9007  10308 11608 12909 14210 15510 16811 18111 19412 20712 22013 23314 24614 25915

budgets=(
53 163 293 423 553 683 813  943  1073
1204 1854 2504 3154  3805 4455 5105 5756 6406 7056
7707 9007  10308 11608 12909 15510 18111 20712 23314 25915
)

# 顺序执行
for b in "${budgets[@]}"; do
    echo "🚀 正在运行: python ../exp7_only_2phase_succreject.py --budget=${b}"
    python ../exp7_only_2phase_succreject.py --dataset=connect --batch_size=64 --max_epochs=16 --budget=${b} --num_workers=4 --device_ids=1 --gpu_ids=1,1,1,1
    echo "✅ budget=${b} 执行完成，等待 3 秒..."
    sleep 3
done

echo "🎉 所有 budget 执行完毕！"
