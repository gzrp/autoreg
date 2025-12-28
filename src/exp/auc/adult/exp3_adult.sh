#!/bin/bash

# 定义 budget 列表
budgets=(
1500 1800 2100
3300 3600 3900
5100 5400 5700
6900 7200 7500
)
k_ns=(
0.05 0.01 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50
)

# 顺序执行
for b in "${budgets[@]}"; do
    for k in "${k_ns[@]}"; do
        echo "🚀 正在运行: python exp2_2phase.py --budget=${b} k_n=${k}"
        python exp3_2phase.py --dataset=adult --batch_size=64 --max_epochs=8 --num_workers=4 --grace_period=1 --budget=${b} --device_ids=0,1 --gpu_ids0,1,0,1 --k_n=
        echo "✅ budget=${b}, k_n=${k} 执行完成，等待 3 秒..."
        sleep 3
    done
done
echo "🎉 所有 budget × k_n 组合执行完毕！"
