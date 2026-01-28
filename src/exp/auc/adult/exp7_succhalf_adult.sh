#!/bin/bash

# 定义 budget 列表
# 8  10 20 30 40 50  60  70  80  90  100 150 200 250 300 350 400  450  500  550  600  700  800  900  1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000
budgets=(
20 25 50 75 99 123 146 172 195 220
245 365 486 608 730 851 972 1095 1215
1337 1460 1701 1945 2189 2430 2675 2918 3160 3403 3645 3890 4131 4375 4618 4860
)

# 顺序执行
for b in "${budgets[@]}"; do
    echo "🚀 正在运行: python ../exp7_only_2phase_succhalf.py --budget=${b}"
    python ../exp2_2phase.py --dataset=adult --batch_size=64 --max_epochs=8 --budget=${b} --num_workers=4 --device_ids=0,1 --gpu_ids=0,1,0,1
    echo "✅ budget=${b} 执行完成，等待 3 秒..."
    sleep 3
done

echo "🎉 所有 budget 执行完毕！"
