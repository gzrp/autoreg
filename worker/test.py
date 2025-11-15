import time

import math


def heavy_task(x: int):
    """CPU 密集型任务"""
    start = time.time()
    s = 0
    for i in range(50_000_000):
        s += math.sqrt((x + i) % 100)

    res = {
        "s": s,
        "time": time.time() - start,
    }
    return res

if __name__ == '__main__':
    start = time.time()
    heavy_task(100)
    print(time.time() - start)