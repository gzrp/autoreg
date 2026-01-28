#!/bin/bash

# 定义 budget 列表
# num          13  20  30  40  50  60  70   80   90    100  150  200   250  300  350  400  450  500  550  600  700   800   900   1000  1100  1200  1300  1400  1500  1600  1700  1800  1900  2000
# succhalf     53  82  122 163 204 244 285 326  366    407  610  813   1017 1220 1423 1626 1829 2033 2236 2439 2846  3252 3658   4065  4471  4878  5284  5691  6097  6503  6910  7316  7723  8129

budgets=(
53  82  122 163 204 244 285 326  366
407  610  813   1017 1220 1423 1626 1829 2033 2236
2439 2846  3252 3658   4065  4471  4878  5284  5691  6097  6503  6910  7316  7723  8129
)

# 顺序执行
for b in "${budgets[@]}"; do
    echo "🚀 正在运行: python ../exp7_only_2phase_succhalf.py --budget=${b}"
    python ../exp7_only_2phase_succhalf.py --dataset=connect --batch_size=64 --max_epochs=16 --budget=${b} --num_workers=4 --device_ids=0 --gpu_ids=0,0,0,0
    echo "✅ budget=${b} 执行完成，等待 3 秒..."
    sleep 3
done

echo "🎉 所有 budget 执行完毕！"
