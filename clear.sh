#!/bin/bash

BASE_DIR="/data/ruipeng/ray_results/bohb"

while true; do
    echo "=================================================="
    echo "[INFO] Cleanup start: $(date)"

    echo "[INFO] Disk usage BEFORE cleanup:"
    du -sh "$BASE_DIR"

    find "$BASE_DIR" -mindepth 1 -maxdepth 1 -type d | while read -r trial_dir; do
        mapfile -t ckpts < <(
            find "$trial_dir" -maxdepth 1 -type d -name "checkpoint_[0-9]*" \
            | awk -F/ '{print $NF}' \
            | sort -V
        )

        # 少于等于 1 个，不处理
        if [ "${#ckpts[@]}" -le 1 ]; then
            continue
        fi

        keep="${ckpts[-1]}"

        echo "[INFO] Trial: $(basename "$trial_dir")"
        echo "       Keep: $keep"

        for ckpt in "${ckpts[@]}"; do
            if [ "$ckpt" != "$keep" ]; then
                echo "       Delete: $ckpt"
                rm -rf "$trial_dir/$ckpt"
            fi
        done
    done

    echo "[INFO] Disk usage AFTER cleanup:"
    du -sh "$BASE_DIR"

    echo "[INFO] Cleanup end: $(date)"
    echo "[INFO] Sleep 20 minutes..."
    echo "=================================================="
    sleep 1200
done
