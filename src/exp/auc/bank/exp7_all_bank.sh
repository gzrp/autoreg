#!/bin/bash

# 定义 budget 列表
# uniform
# 4  10  20  30  40  50  60  70   80   90    100  150  200   250  300  350  400  450  500  550  600  700   800   900   1000  1200  1400  1600  1800  2000
# 32 80  159 238 317 396 475 554 633  712    791  1187 582   1978 2373 2768 3164 3559 3955 4350 4746 5536  6327  7118  7909  9491  11072 12654 14236 15817

budgets=(
15817
)


# 顺序执行
for b in "${budgets[@]}"; do
    echo "🚀 正在运行: python ../exp7_only_2phase_all.py --budget=${b}"
    python ../exp7_only_2phase_all.py --dataset=bank --batch_size=64 --max_epochs=16 --budget=${b} --num_workers=4 --device_ids=0 --gpu_ids=0
    echo "✅ budget=${b} 执行完成，等待 3 秒..."
    sleep 3
done

echo "🎉 所有 budget 执行完毕！"
