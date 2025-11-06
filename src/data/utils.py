import numpy as np
from typing import List, Literal


def compute_class_weights(
        class_counts: List[int],
        method: Literal["inv", "power", "log"] = "inv",
        normalize: bool = False,
        alpha: float = 0.5,
        mu: float = 1.5,
        clip_min: float = 0.1,
        clip_max: float = 30.0,
) -> List[float]:
    """
    计算类别权重
    - "inv":   反频率  w_i = N/(K*n_i)
    - "power": 幂次平滑 w_i = (N/(K*n_i))**alpha
    - "log":   对数压缩 w_i = log(mu * N / n_i)
    :param class_counts: 样本数量
    :param method: 使用方法
    :param normalize: 是否规划化
    :param alpha: 幂次平滑指数，默认 0.5
    :param mu: 对数压缩强度
    :param clip_min: 裁剪下限
    :param clip_max: 裁剪上线
    :return:
    """
    counts = np.asarray(class_counts, dtype=float)
    safe_counts = np.where(counts > 0.0, counts, 1.0)
    K = float(len(safe_counts))
    N = float(safe_counts.sum())

    # 计算基础权重
    if method == "inv":
        w = N / (K * safe_counts)
    elif method == "power":
        w = (N / (K * safe_counts)) ** alpha
    elif method == "log":
        # log 压缩：自变量需 >0
        val = mu * N / safe_counts
        w = np.log(np.maximum(val, 1e-6))
    else:
        raise ValueError(f"Unknown method: {method}")

    # 统一权重裁剪（先下限后上限）
    if clip_min is not None:
        w = np.maximum(w, clip_min)
    if clip_max is not None:
        w = np.minimum(w, clip_max)

    # 归一化到均值=1（不改变相对比例）
    if normalize:
        s = w.sum()
        if s > 0:
            w = w * (len(w) / s)

    return w.tolist()