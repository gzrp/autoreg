#!/bin/bash

# 定义 budget 列表
#  4  10 20  30  40  50  60  70  80  90  100 150 200 250  300  350  400  450  500  550  600  700  800  900  1000 1200 1400 1600 1800 2000
budgets=(
#20 50 100 150 195 245 295 345 390 440
#490 730 975 1215 1460
1705 1945 2190 2430 2675
2920 3405 3890 4375 4860 5835 6805 7780 8750 9720
)

# 顺序执行
for b in "${budgets[@]}"; do
    echo "🚀 正在运行: python ../exp7_only_2phase_uniform.py --budget=${b}"
    python ../exp7_only_2phase_uniform.py --dataset=adult --batch_size=64 --max_epochs=8 --budget=${b} --num_workers=4 --device_ids=2,2 --gpu_ids=2,2,2,2
    echo "✅ budget=${b} 执行完成，等待 3 秒..."
    sleep 3
done

echo "🎉 所有 budget 执行完毕！"
