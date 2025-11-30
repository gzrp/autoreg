import json

dataset = "dionis"

def func11():
    input_file = f"/data/ruipeng/workdir/autoreg/.raw/exp2/{dataset}/all/asha_time_log.jsonl"      # 原始 JSONL 文件路径
    output_file = f"/data/ruipeng/workdir/autoreg/.raw/exp2/{dataset}/extract/asha_time_log_extract.jsonl"    # 输出文件路径

    with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            data = json.loads(line.strip())  # 每行是一个 JSON 对象
            # 提取需要的字段
            result = {
                "elapsed_time": data.get("elapsed_time"),
                "best_metric": data.get("best_metric"),
                "val_bacc_history_max": max(data.get("val_bacc_history")),
                "val_bacc_history": data.get("val_bacc_history"),
            }
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    print("✅ 提取完成，结果已保存到", output_file)

def func12():
    input_file = f"/data/ruipeng/workdir/autoreg/.raw/exp2/{dataset}/all/hyperband_time_log.jsonl"      # 原始 JSONL 文件路径
    output_file = f"/data/ruipeng/workdir/autoreg/.raw/exp2/{dataset}/extract/hyperband_time_log_extract.jsonl"    # 输出文件路径

    with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            data = json.loads(line.strip())  # 每行是一个 JSON 对象
            # 提取需要的字段
            result = {
                "elapsed_time": data.get("elapsed_time"),
                "best_metric": data.get("best_metric"),
                "val_bacc_history_max": max(data.get("val_bacc_history")),
                "val_bacc_history": data.get("val_bacc_history"),
            }
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    print("✅ 提取完成，结果已保存到", output_file)\

def func13():
    input_file = f"/data/ruipeng/workdir/autoreg/.raw/exp2/{dataset}/all/bohb_time_log.jsonl"      # 原始 JSONL 文件路径
    output_file = f"/data/ruipeng/workdir/autoreg/.raw/exp2/{dataset}/extract/bohb_time_log_extract.jsonl"    # 输出文件路径

    with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            data = json.loads(line.strip())  # 每行是一个 JSON 对象
            # 提取需要的字段
            result = {
                "elapsed_time": data.get("elapsed_time"),
                "best_metric": data.get("best_metric"),
                "val_bacc_history_max": max(data.get("val_bacc_history")),
                "val_bacc_history": data.get("val_bacc_history"),
            }
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    print("✅ 提取完成，结果已保存到", output_file)


def func2():
    input_file = f"/data/ruipeng/workdir/autoreg/.raw/exp2/{dataset}/all/2phase_time_log.jsonl"  # 原始 JSONL 文件路径
    output_file = f"/data/ruipeng/workdir/autoreg/.raw/exp2/{dataset}/extract/2phase_time_log_extract.jsonl"  # 输出文件路径

    with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            if not line.strip():
                continue
            data = json.loads(line.strip())

            elapsed_time = data.get("Budget", 0)
            best_exploit = data.get("best_exploit")
            if best_exploit is not None and isinstance(best_exploit, dict):
                best_metric = best_exploit.get("bacc", 0.0)
                val_bacc_history = best_exploit.get("bacc_history")

            else:
                best_metric = 0.0
                val_bacc_history = []
            # 计算最大值
            bacc_history_max = max(val_bacc_history) if val_bacc_history else 0.0
            result = {
                "elapsed_time": elapsed_time,
                "best_metric": best_metric,
                "val_bacc_history_max": bacc_history_max,
                "val_bacc_history": val_bacc_history,
            }
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    print("✅ 提取完成，结果已保存到", output_file)


def func3():
    input_file = f"/data/ruipeng/workdir/autoreg/.raw/exp2/{dataset}/all/1phase_time_log.jsonl"  # 原始 JSONL 文件路径
    output_file = f"/data/ruipeng/workdir/autoreg/.raw/exp2/{dataset}/extract/1phase_time_log_extract.jsonl"  # 输出文件路径

    with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            if not line.strip():
                continue
            data = json.loads(line.strip())

            elapsed_time = data.get("Budget", 0)
            # 取 train_result.bacc
            train_result = data.get("train_result")
            if train_result is not None and isinstance(train_result, dict):
                best_metric = train_result.get("bacc", 0.0)
                bacc_history = train_result.get("val_bacc_history", [])
            else:
                best_metric = 0.0
                bacc_history = []
            # 计算最大值
            bacc_history_max = max(bacc_history) if bacc_history else 0.0
            result = {
                "elapsed_time": elapsed_time,
                "best_metric": best_metric,
                "val_bacc_history_max": bacc_history_max,
                "val_bacc_history": bacc_history,
            }
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    print("✅ 提取完成，结果已保存到", output_file)



if __name__ == '__main__':
    func13()