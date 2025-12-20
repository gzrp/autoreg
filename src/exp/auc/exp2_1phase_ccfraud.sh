#!/bin/bash

# 定义 budget 列表
budgets=(
1
)

# 顺序执行
for b in "${budgets[@]}"; do
    echo "🚀 正在运行: python exp2_1phase.py --budget=${b}"
    python exp2_1phase.py --dataset=bank --batch_size=64 --max_epochs=16 --grace_period=1 --budget=${b} --device_ids=0,1
    echo "✅ budget=${b} 执行完成，等待 5 秒..."
    sleep 5
done

echo "🎉 所有 budget 执行完毕！"
