import zmq
import pickle
import time

WORKERS = [
    {"id": "w1", "port": 6001},
    {"id": "w2", "port": 6002},
    {"id": "w3", "port": 6003},
    {"id": "w4", "port": 6004},
]

def main():
    ctx = zmq.Context()
    # 为每个 worker 建立一个 REQ socket
    sockets = {}
    for w in WORKERS:
        sock = ctx.socket(zmq.REQ)
        sock.connect(f"tcp://127.0.0.1:{w['port']}")
        sockets[w['id']] = sock
        print(f"已连接 {w['id']} at port {w['port']}")

    print("\n开始发送任务...\n")
    # 给每个 worker 发送任务
    start = time.time()
    for i, w in enumerate(WORKERS):
        x = 100 + i  # 测试任务
        print(f"向 {w['id']} 发送任务 x={x}")
        sockets[w['id']].send(pickle.dumps({"x": x}))

    # 接收每个 worker 的结果
    for w in WORKERS:
        reply = sockets[w['id']].recv()
        data = pickle.loads(reply)
        print(f"{w['id']} 返回结果: result={data['result']}  (core={data['core']})")
    print(f"使用时间 {time.time() - start} seconds")

    print("\n所有任务完成！")

if __name__ == '__main__':
    main()