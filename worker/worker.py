import sys
import zmq
import psutil
import os
import time
import pickle
import math


def bind_cpu(worker_id: int, core_id: int):
    """将当前进程绑定到指定 CPU 核"""
    p = psutil.Process(os.getpid())
    p.cpu_affinity([core_id])
    print(f"[Worker {worker_id}] CPU 亲和性设置为核心 {core_id}")

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

def start_worker(worker_id: int, core_id: int, port: int):
    bind_cpu(worker_id, core_id)
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REP)
    socket.bind(f"tcp://127.0.0.1:{port}")
    print(f"[Worker {worker_id}] 已启动，核心={core_id} 端口={port}\n")
    while True:
        msg = socket.recv()
        data = pickle.loads(msg)
        x = data["x"]
        print(f"[Worker {worker_id}] heavy_task({x}) 开始计算...")

        result = heavy_task(x)

        print(f"[Worker {worker_id}] heavy_task({x}) 完成！发送结果。\n")
        socket.send(pickle.dumps({
            "result": result,
            "core": core_id,
            "worker": worker_id
        }))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("用法: python worker.py <worker_id> <core_id> <port>")
        sys.exit(1)

    worker_id = sys.argv[1]
    core_id = int(sys.argv[2])
    port = int(sys.argv[3])

    start_worker(worker_id, core_id, port)

