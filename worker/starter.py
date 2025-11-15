import subprocess
import time
import sys

# 启动的 Worker 配置
WORKER_SCRIPT = "worker.py"
WORKERS = [
    {"id": "w1", "core": 31, "port": 6001},
    {"id": "w2", "core": 32, "port": 6002},
    {"id": "w3", "core": 33, "port": 6003},
    {"id": "w4", "core": 34, "port": 6004},
]

def main():
    procs = []
    for w in WORKERS:
        print(f"启动 Worker {w['id']} (core={w['core']}, port={w['port']}) ...")
        # 启动 worker.py 的子进程
        p = subprocess.Popen(
            [
                sys.executable,
                WORKER_SCRIPT,
                w["id"],
                str(w["core"]),
                str(w["port"])
            ]
        )
        procs.append(p)
        time.sleep(0.3)  # 小延迟避免日志重叠

    print("\n所有 worker 已全部启动！")
    print("按 Ctrl + C 停止所有 worker。\n")

    try:
        # 保持主脚本不退出
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        print("\n收到中断信号，正在终止 workers ...")
        for p in procs:
            p.terminate()
        print("所有 worker 已停止。")

if __name__ == '__main__':
    main()