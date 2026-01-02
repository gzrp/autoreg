#!/bin/bash

configs=(
"8 4 0.5 0,1,2,3 0,1,2,3,0,1,2,3"
"4 2 0.5 2,3 2,3,2,3"
"2 1 0.5 2 2,2"
"1 1 1 2 2"
)

for cfg in "${configs[@]}"; do
    read nw ng tng d g<<< "$cfg"
    echo "🚀 正在运行: python exp4_2phase.py --num_workers=${n} --device_ids=${d} --gpu_ids=${g}"
    python ../exp4_2phase.py --dataset=adult --batch_size=64 --max_epochs=8 --num_samples=2000 --k_n=0.25 --num_workers=${nw} --num_gpus=${ng} --trail_num_gpus=${tng} --device_ids=${d} --gpu_ids=${g}
    sleep 3
done
echo "🎉 所有 budget × k_n 组合执行完毕！"