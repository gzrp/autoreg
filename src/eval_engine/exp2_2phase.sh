#!/bin/bash

# 定义 budget 列表
budgets=(
2400 2596
)

# 顺序执行
for b in "${budgets[@]}"; do
    echo "🚀 正在运行: python reg_selection_time.py --budget=${b}"
    python reg_selection_time.py --dataset=diabetic --batch_size=64 --max_epochs=4 --grace_period=2 --budget=${b}
    echo "✅ budget=${b} 执行完成，等待 5 秒..."
    sleep 5
done

echo "🎉 所有 budget 执行完毕！"
