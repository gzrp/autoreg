import time

if __name__ == '__main__':

    print("开始测试 PyCharm 控制台循环缓冲区（100MB）")
    print("请观察控制台是否会在达到 100MB 左右时自动丢弃旧输出。\n")

    # 每行大约 100 字节，100万行 ≈ 100MB
    for i in range(4_000_000):  # 输出约 200MB
        print(f"Line {i:08d}: 这是一个测试输出，目的是填满 PyCharm 的控制台日志缓冲区。")
        if i % 100000 == 0:
            time.sleep(0.1)  # 让 PyCharm 有时间刷新显示

    print("\n测试结束，请检查控制台开头的输出是否已经被自动丢弃。")
