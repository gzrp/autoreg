#!/bin/bash

# 定义 budget 列表
# uniform
# 4   10  20   30   40   50   60   70   80    90    100  150  200   250   300   350   400   450   500   550   600   700   800   900   1000  1200  1400  1600  1800   2000
# 244 610 1220 1829 2439 3048 3658 4267 4877 5486   6096 9143 12191 15239 18286 21334 24382 27429 30477 33525 36572 42667 48763 54858 60953 73144 85334 97525 109716 121906
budgets=(
121906
)

# 顺序执行
for b in "${budgets[@]}"; do
    echo "🚀 正在运行: python ../exp7_only_2phase_all.py --budget=${b}"
    python ../exp7_only_2phase_all.py --dataset=frappe --batch_size=256 --max_epochs=16 --budget=${b} --num_workers=4 --device_ids=0 --gpu_ids=0
    echo "✅ budget=${b} 执行完成，等待 3 秒..."
    sleep 3
done

echo "🎉 所有 budget 执行完毕！"
