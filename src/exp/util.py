import json
import os
from datetime import datetime

import pandas as pd
import torch
import numpy as np
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_results(df: pd.DataFrame):
    # 按 bacc 降序排序
    df_sorted = df.sort_values(by="bacc", ascending=False)
    items = []
    for _, row in df_sorted.iterrows():
        cfg = {}
        for col in df_sorted.columns:
            if col.startswith("config/"):
                val = row[col]
                # 将 numpy 字符串转回普通 str
                if hasattr(val, 'item'):
                    val = val.item()
                cfg[col.replace("config/", "")] = val

        items.append({
            "loss": row["loss"],
            "acc": row["acc"],
            "bacc": row["bacc"],
            "training_iteration": row["training_iteration"],
            "trial_id": row["trial_id"],
            "date": row["date"],
            "time_this_iter_s": row["time_this_iter_s"],
            "time_total_s": row["time_total_s"],
            "config": cfg
        })
    return items


def get_top_k(results, K_N :float = 0.2):
    if not results:
        return []
    # 按 score 降序排序
    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
    # 计算 Top-K 数量
    k = max(1, int(len(sorted_results) * K_N))
    return sorted_results[:k]


def save_results_json(results: dict, exp_name: str, output_dir: str = None):
    if output_dir is None:
        output_dir = os.getcwd()  # 当前工作目录

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{exp_name}_{timestamp}.json"
    json_path = os.path.join(output_dir, filename)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"✅ 实验结果已保存到: {json_path}")
    return json_path


def numpy_to_python(obj):
    if isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(v) for v in obj]
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.str_):
        return str(obj)
    else:
        return obj