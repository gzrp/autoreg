#!/bin/bash

configs=(
"8 0,1,2,3 0,1,2,3,0,1,2,3"
"4 2,3, 2,3,2,3"
"2 2 2,2"
"1 2 2"
)

for cfg in "${configs[@]}"; do
    read n d g <<< "$cfg"
    echo "🚀 正在运行: python exp4_2phase.py --num_workers=${n} --device_ids=${d} --gpu_ids=${g}"
    python ../exp4_2phase.py --dataset=adult --batch_size=64 --max_epochs=8 --num_samples=2000 --num_workers=${n} --grace_period=1 --device_ids=${d} --gpu_ids=${g}
    sleep 3
done
echo "🎉 所有 budget × k_n 组合执行完毕！"