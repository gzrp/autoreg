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

    return {"task_id": x, "sum": s, "cost": time.time() - start}


def main():
    pool_size = len(CPU_CORES)
    print(f"创建进程池，共 {pool_size} 个 worker，每个 worker 绑定一个 CPU 核\n")
    pool = Pool(processes=pool_size, initializer=init_worker)
    TOTAL_TASKS = 200
    results = []
    state = {"next_task": 0, "finished": 0}

    def task_done(res):
        """任务完成自动调用"""
        results.append(res)
        state["finished"] += 1
        print(f"[完成] 任务 {res['task_id']}，耗时 {res['cost']:.2f} 秒")

        # 如果还有剩余任务，继续提交
        if state["next_task"] < TOTAL_TASKS:
            tid = state["next_task"]
            print(f"提交任务 {tid}")
            pool.apply_async(heavy_task, args=(tid,), callback=task_done)
            state["next_task"] += 1

    start = time.time()
    print("\n提交初始任务...\n")
    for _ in range(pool_size):
        tid = state["next_task"]
        print(f"提交任务 {tid}")
        pool.apply_async(heavy_task, args=(tid,), callback=task_done)
        state["next_task"] += 1

    # 等待所有任务完成
    while state["finished"] < TOTAL_TASKS:
        time.sleep(0.1)

    pool.close()
    pool.join()
    usage = time.time() - start
    print(f"时间：{usage}, 平均时间 {usage/TOTAL_TASKS}")
    print("\n全部任务完成！最后 3 个结果：")
    for r in results[-3:]:
        print(r)

if __name__ == "__main__":
    main()