#!/bin/bash

# 定义 budget 列表
budgets=(
500 1000 1500 2000
)

# 顺序执行
for b in "${budgets[@]}"; do
    echo "🚀 正在运行: python exp6_2phase_cache.py --num_samples=${b}"
    python ../exp6_2phase_cache.py --dataset=frappe --batch_size=256 --max_epochs=16 --num_samples=${b} --k_n=0.2 --sample_ratio=0.2 --num_workers=1 --device_ids=0 --gpu_ids=0
    echo "✅ budget=${b} 执行完成，等待 5 秒..."
    sleep 5
done

echo "🎉 所有 budget 执行完毕！"
