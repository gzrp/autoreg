


import re

if __name__ == '__main__':

    with open("/home/zrp/pycharmProjects/autoreg/.exp/experiment_state-2025-11-11_02-32-30.json", "r", encoding="utf-8") as f:
        text = f.read()
    #
    # baccs = re.findall(r'"bacc"\s*:\s*([0-9]+\.[0-9]+)', text)
    # print("\n".join(baccs))
    #
    # # 可选：保存到文件
    # with open("bacc_values.txt", "w", encoding="utf-8") as out:
    #     out.write("\n".join(baccs))

    # 读取文件
    # with open("your_file.txt", "r", encoding="utf-8") as f:
    #     text = f.read()

    # 针对 \"bacc\": 0.8216 这种模式的匹配
    pattern = r'\\"bacc\\"\s*:\s*([0-9]+\.[0-9]+)'

    matches = re.findall(pattern, text)
    print(f"共提取到 {len(matches)} 个 bacc 值：")
    print("\n".join(matches))

    # 保存结果
    with open("bacc_values.txt", "w", encoding="utf-8") as out:
        out.write("\n".join(matches))