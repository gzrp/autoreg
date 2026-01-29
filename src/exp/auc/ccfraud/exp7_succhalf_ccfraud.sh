#!/bin/bash

# 定义 budget 列表
# 12 20  30  40  50  60  70  80  90
# 100 150 200   250  300  350  400  450  500  550
# 600  700  800  900  1000 1200 1400 1600 1800 2000
budgets=(
66 103 154 205 256 307 358 409 460
511 766 1022 1277 1532 1788 2043  2298 2554 2809
3064 3575 4085 4596 5107 6128 7149 8170 9192 10213
)

# 顺序执行
for b in "${budgets[@]}"; do
    echo "🚀 正在运行: python ../exp7_only_2phase_succhalf.py --budget=${b}"
    python ../exp7_only_2phase_succhalf.py --dataset=ccfraud --batch_size=256 --max_epochs=16 --budget=${b} --num_workers=4 --device_ids=0,1 --gpu_ids=0,1,0,1
    echo "✅ budget=${b} 执行完成，等待 3 秒..."
    sleep 3
done

echo "🎉 所有 budget 执行完毕！"
