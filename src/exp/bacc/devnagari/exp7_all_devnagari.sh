#!/bin/bash

# 定义 budget 列表
# uniform
# 4  10  20   30   40   50   60   70   80    90    100   150  200   250   300   350   400   450   500   550   600   700   800   900   1000  1200  1400  1600  1800    2000
# 74 184 368  552 735  919  1103 1286 1470 1654   1837  2756  3674 4593  5511  6430  7348  8266  9185  10103 11022 12859  14696 16532 18369 22043 25717 29391 33064  36738

budgets=(
36738
)

# 顺序执行
for b in "${budgets[@]}"; do
    echo "🚀 正在运行: python ../exp7_only_2phase_all.py --budget=${b}"
    python ../exp7_only_2phase_all.py --dataset=devnagari --batch_size=64 --max_epochs=16 --budget=${b} --num_workers=4 --device_ids=2 --gpu_ids=2
    echo "✅ budget=${b} 执行完成，等待 3 秒..."
    sleep 3
done

echo "🎉 所有 budget 执行完毕！"
q