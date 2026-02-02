#!/bin/bash

# 定义 budget 列表
# num        8  10 20 30 40 50  60  70  80  90  100 150 200 250 300 350 400  450  500  550  600  700  800  900  1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000
# succhalf   20 25 49 73 98 122 146 171 195 219 243 365 486 608 729 851 972 1095 1215 1337 1458 1701 1944  2187 2430 2673 2916 3159 3402 3645 3888 4131 4374 4617 4860
budgets=(
20 25 49 73 98 122 146 171 195 219 243 365 486 608 729 851 972 1095 1215 1337 1458 1701 1944  2187 2430 2673 2916 3159 3402 3645 3888 4131 4374 4617 4860
)

# 顺序执行
for b in "${budgets[@]}"; do
    echo "🚀 正在运行: python ../exp7_only_2phase_succhalf.py --budget=${b}"
    python ../exp7_only_2phase_succhalf.py --dataset=adult --batch_size=64 --max_epochs=8 --budget=${b} --num_workers=4 --device_ids=0 --gpu_ids=0
    echo "✅ budget=${b} 执行完成，等待 1 秒..."
    sleep 1
done

echo "🎉 所有 budget 执行完毕！"
