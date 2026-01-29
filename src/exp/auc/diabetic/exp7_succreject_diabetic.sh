#!/bin/bash

# 定义 budget 列表
#
# 11  20  30  40  50  60  70   80   90
# 100  150  200   250  300  350  400  450  500  550
# 600  700   800   900   1000   1200   1400   1600   1800   2000
# succrejct    59 183 329 476 622 768 914  1061 1207   1353 2084 2815 3546  4277 5008 5739 6471 7202 7933 8664 10126 11588 13050 14512 15975 17437 18899 20361 21823 23285 24748 26210 27672 29134

budgets=(
59 183 329 476 622 768 914  1061 1207
1353 2084 2815 3546  4277 5008 5739 6471 7202 7933
8664 10126 11588 13050 14512 17437 20361 23285 26210 29134
)

# 顺序执行
for b in "${budgets[@]}"; do
    echo "🚀 正在运行: python ../exp7_only_2phase_succreject.py --budget=${b}"
    python ../exp7_only_2phase_succreject.py --dataset=diabetic --batch_size=128 --max_epochs=16 --budget=${b} --num_workers=4 --device_ids=0,1 --gpu_ids=0,1,0,1
    echo "✅ budget=${b} 执行完成，等待 3 秒..."
    sleep 3
done

echo "🎉 所有 budget 执行完毕！"
