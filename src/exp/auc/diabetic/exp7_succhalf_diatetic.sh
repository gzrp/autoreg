#!/bin/bash

# 定义 budget 列表
# 12  20  30  40  50  60  70   80   90
# 100  150  200   250  300  350  400  450  500  550
# 600  700   800   900   1000   1200   1400   1600   1800   2000
# succhalf     59  92 138 183 229 275 320  366  412    457  686  914   1143 1371 1600 1828 2057 2285 2514 2742 3199 3656  4113   4570  5027  5484  5941  6397  6854  7311  7768  8225  8682  9139

budgets=(
59  92 138 183 229 275 320  366  412
457  686  914   1143 1371 1600 1828 2057 2285 2514
2742 3199 3656  4113   4570   5484   6397   7311   8225   9139
)

# 顺序执行
for b in "${budgets[@]}"; do
    echo "🚀 正在运行: python ../exp7_only_2phase_succhalf.py --budget=${b}"
    python ../exp7_only_2phase_succhalf.py --dataset=diabetic --batch_size=128 --max_epochs=16 --budget=${b} --num_workers=4 --device_ids=0,1 --gpu_ids=0,1,0,1
    echo "✅ budget=${b} 执行完成，等待 3 秒..."
    sleep 3
done

echo "🎉 所有 budget 执行完毕！"
