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
    next_task_id = 0          # 下一个要提交的任务编号
    finished_tasks = 0        # 完成任务计数
    results = []              # 存储所有结果
    pending = []              # 正在执行中的 async 任务句柄
    TOTAL_TASKS = 200

    # ========================================================
    # 1) 先提交 worker 数量个任务
    # ========================================================
    start = time.time()
    print("提交初始任务...\n")
    for _ in range(pool_size):
        print(f"提交任务 {next_task_id}")
        pending.append(pool.apply_async(heavy_task, args=(next_task_id,)))
        next_task_id += 1

    # ========================================================
    # 2) 动态补充任务 —— 谁完成，就继续提交下一个任务
    # ========================================================
    while finished_tasks < TOTAL_TASKS:
        for p in pending[:]:  # 遍历复制的列表
            if p.ready():  # 该任务完成
                res = p.get()
                results.append(res)
                finished_tasks += 1
                pending.remove(p)
                print(f"[完成] 任务 {res['task_id']} ，耗时 {res['cost']:.2f} 秒")
                # 如果还有任务没提交，就补一个新的
                if next_task_id < TOTAL_TASKS:
                    print(f"提交任务 {next_task_id}")
                    pending.append(pool.apply_async(heavy_task, args=(next_task_id,)))
                    next_task_id += 1

        time.sleep(0.1)  # 避免循环过快

    # ========================================================
    # 3) 所有任务完成
    # ========================================================
    pool.close()
    pool.join()

    print("\n=====================================")
    total_time = time.time() - start
    print(f"全部 {TOTAL_TASKS} 个任务完成！时间：{total_time}, 平均时间 {total_time / TOTAL_TASKS}")
    print("最后 5 个任务结果：")
    for r in results[-5:]:
        print(r)
    print("=====================================\n")


if __name__ == "__main__":
    main()
    start = time.time()
    for i in range(200):
        res = heavy_task(i)
        print(res)
    print((time.time() - start) / 200)