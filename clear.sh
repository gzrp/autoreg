#!/bin/bash

BASE_DIR="/data/ruipeng/ray_results/hyperband"

echo "[INFO] Cleanup start: $(date)"

find "$BASE_DIR" -mindepth 1 -maxdepth 1 -type d | while read -r trial_dir; do
    # 找到该 trial 下所有 checkpoint 目录并排序
    mapfile -t ckpts < <(
        find "$trial_dir" -maxdepth 1 -type d -name "checkpoint_[0-9]*" \
        | awk -F/ '{print $NF}' \
        | sort -V
    )

    # 少于等于 2 个，不处理
    if [ "${#ckpts[@]}" -le 1 ]; then
        continue
    fi
    # 编号最大的那个
    keep="${ckpts[-1]}"

    echo "[INFO] Trial: $(basename "$trial_dir")"
    echo "       Keep: $keep"

    # 删除其余
    for ckpt in "${ckpts[@]}"; do
        if [ "$ckpt" != "$keep" ]; then
            echo "       Delete: $ckpt"
            rm -rf "$trial_dir/$ckpt"
        fi
    done
done
