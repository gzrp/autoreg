#!/bin/bash

# 定义 budget 列表
budgets=(
500 1000 1500 2000
)

# 顺序执行
for b in "${budgets[@]}"; do
    echo "🚀 正在运行: python exp6_1phase_cache.py --budget=${b}"
    python ../exp6_1phase_cache.py --dataset=adult --batch_size=64 --num_samples=${b} --num_workers=1 --device_ids=3 --gpu_ids=3
    echo "✅ budget=${b} 执行完成，等待 5 秒..."
    sleep 5
done

echo "🎉 所有 budget 执行完毕！"
