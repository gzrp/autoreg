import os
import psutil
import math
import time
from multiprocessing import Pool, current_process

# ==========================
# 配置区域：你的 CPU 核心编号
# ==========================
CPU_CORES = [31, 32, 33, 34]   # 这里写你要绑定的 CPU 核编号，比如你的大核


def init_worker():
    """进程启动时执行：根据进程号绑定 CPU"""
    wid = current_process()._identity[0] - 1

    # 当前 worker 的 CPU 核编号
    core = CPU_CORES[wid]
    p = psutil.Process(os.getpid())
    p.cpu_affinity([core])

    print(f"[Worker PID={os.getpid()}] 绑定 CPU 核 {core}")


def heavy_task(x: int):
    """CPU 密集型任务"""
    start = time.time()
    s = 0
    for i in range(30_000_000):
        s += math.sqrt((x + i) % 100)

    res = {
        "s": s,
        "time": time.time() - start,
    }
    return res


def main():
    print("创建进程池并设置 CPU 亲和性...\n")
    pool = Pool(
        processes=len(CPU_CORES),  # 多少核就开多少个 worker
        initializer=init_worker  # 每个 worker 启动时绑定 CPU
    )

    start = time.time()
    print("提交任务中...\n")
    tasks = list(range(8))  # 提交 8 个任务
    results = pool.map(heavy_task, tasks)

    print("\n所有任务已完成！结果如下：")
    for r in results:
        print(r)
    print(f"花费时间{time.time()-start}")
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
