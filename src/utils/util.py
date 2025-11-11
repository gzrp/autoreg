
import json
from datetime import datetime
import os

import numpy as np


def save_dict_to_file(data: dict, base_dir: str = "./output", prefix: str = "dict"):
    """
    将字典保存为 JSON 文件，目录名为当天日期（YYYYMMDD），文件名包含时间戳（HHMMSS）。

    参数:
        data (dict): 要保存的字典
        base_dir (str): 基础保存目录，默认为 ./output
        prefix (str): 文件名前缀，默认为 "dict"

    返回:
        str: 保存的文件完整路径
    """
    # 获取当前日期和时间
    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M%S")

    # 构建目录路径
    save_dir = os.path.join(base_dir, date_str)
    os.makedirs(save_dir, exist_ok=True)

    # 生成文件名
    filename = f"{prefix}_{time_str}.json"
    filepath = os.path.join(save_dir, filename)

    # 保存为 JSON 文件
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"✅ 字典已保存到: {filepath}")
    return filepath


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