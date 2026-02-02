import argparse
import copy
import math
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Any
from torch.utils.data import DataLoader
from src.data.datasets import get_dataset
from src.data.meta import get_metadata
from src.data.utils import compute_class_weights
from src.model.backbone import BackboneMLP
from src.profiling.profiling import get_profile_data
from src.searcher.random_searcher import RandomSearcher
from src.trainer.trainer_new import Trainer
from src.space.space import reg_space
from src.utils.util import numpy_to_python, append_jsonl, save_dict_to_file
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

# 全局缓存 META & DATASET
GLOBAL_DATASET_META = None
GLOBAL_TRAIN_SET = None
GLOBAL_VALID_SET = None
GLOBAL_TEST_SET = None

def init_global_dataset(args):
    global GLOBAL_DATASET_META, GLOBAL_TRAIN_SET, GLOBAL_VALID_SET, GLOBAL_TEST_SET
    if GLOBAL_DATASET_META is not None:
        return
    start_t = time.time()
    meta = get_metadata(dataset=args.dataset)
    data_dir = meta["data_dir"]
    train_set, valid_set, test_set = get_dataset(dataset=args.dataset, data_dir=data_dir,)
    GLOBAL_DATASET_META = meta
    GLOBAL_TRAIN_SET = train_set
    GLOBAL_VALID_SET = valid_set
    GLOBAL_TEST_SET = test_set
    print(
        f"✅ 进程 {os.getpid()} 首次加载采样后的 Dataset（TRAIN/VALID/TEST），"
        f"耗时：{time.time() - start_t:.3f}s",
        flush=True
    )

class GPUWorker(mp.Process):
    """每个 GPU 一个常驻进程，在队列中循环取 cfg 执行 evaluate"""
    def __init__(self, gpu_id: int, args, task_queue: mp.Queue, result_queue: mp.Queue):
        super().__init__()
        self.gpu_id = gpu_id
        # 每个进程使用 args 的独立拷贝，避免交叉修改
        self.args = copy.copy(args)
        self.task_queue = task_queue
        self.result_queue = result_queue
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        self.args.device = "cuda:0"
        seed = args.seed
        random.seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.manual_seed(seed)
            torch.cuda.set_device(0)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def run(self):
        # 在当前进程里初始化 evaluator，只做一次
        seed = self.args.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        evaluator = ExploitEvaluator(self.args)

        while True:
            cfg = self.task_queue.get()
            if cfg is None:
                # 收到结束信号
                break
            metrics = evaluator.evaluate(cfg)
            # 把结果放回主进程
            self.result_queue.put((cfg, metrics, self.gpu_id))

class ExploitEvaluator:
    def __init__(self, args):
        self.args = args
        self.seed = args.seed
        self.device = args.device

        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.swa_start_epoch = args.swa_start_epoch
        self.max_epochs = args.max_epochs
        self.verbose = args.verbose

        init_global_dataset(args)
        # ===== 使用全局 META/DATASET，而不是每次重新 get_* =====
        global GLOBAL_DATASET_META, GLOBAL_TRAIN_SET, GLOBAL_VALID_SET, GLOBAL_TEST_SET

        if GLOBAL_DATASET_META is not None:
            self.meta = GLOBAL_DATASET_META
        else:
            self.meta = get_metadata(dataset=self.dataset)

        # self.meta = get_metadata(dataset=self.dataset)
        self.in_features = self.meta["in_features"]
        self.out_features = self.meta["out_features"]
        self.hidden_features = [512, 512, 512, 512, 512, 512]
        self.is_balanced = self.meta["is_balanced"]
        self.class_ratio = self.meta["class_ratio"]
        self.data_dir = self.meta["data_dir"]

        # ⭐ 如果全局 DATASET 已初始化，则在当前进程中构建 DataLoader
        if GLOBAL_TRAIN_SET is not None and GLOBAL_VALID_SET is not None and GLOBAL_TEST_SET is not None:
            self.train_set = GLOBAL_TRAIN_SET
            self.valid_set = GLOBAL_VALID_SET
            self.test_set = GLOBAL_TEST_SET
        else:
            # 兜底逻辑：如果没初始化全局数据，保持原始行为
            print("如果没初始化全局数据，保持原始行为")

    def evaluate(self, config) -> dict[str, Any]:
        start_time = time.time()
        # 初始化模型
        model = BackboneMLP(
            input_dim=self.in_features,
            hidden_dims=self.hidden_features,
            output_dim=self.out_features,
            reg_config=config,
        )

        train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False)
        valid_loader = DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)

        ce_weight = None
        if not self.is_balanced:
            weights = compute_class_weights(self.class_ratio, method="inv")
            ce_weight = torch.tensor(weights, dtype=torch.float32).to(torch.device(self.device))

        criterion = nn.CrossEntropyLoss(weight=ce_weight)
        # 初始化训练器
        trainer = Trainer(
            model=model,
            criterion=criterion,
            lr=self.args.lr,
            swa_start_epoch=self.swa_start_epoch,
            device=self.device,
            reg_config=config,
            metric_type="AUC",
        )
        acc_max = 0
        auc_max = 0
        metrics = None
        train_epochs = config["train_epochs"]
        auc_history = []
        loss_history = []
        for epoch in range(train_epochs):
            trainer.train(train_loader, valid_loader, epochs=1, verbose=self.args.verbose)
            loss, acc, auc = trainer.evaluate(test_loader)
            auc_history.append(auc)
            loss_history.append(loss)
            acc_max = max(acc_max, acc)
            auc_max = max(auc_max, auc)
            metrics = {
                "loss": loss,
                "acc": acc_max,
                "auc": auc_max,
                "auc_history": auc_history,
                "loss_history": loss_history,
                "time": time.time() - start_time,
            }
        torch.cuda.synchronize()  # 让 CUDA 异步执行完
        del model
        del trainer
        del criterion
        torch.cuda.empty_cache()
        return metrics

class ExploitPhaseParallel:
    def __init__(self, args, configs):
        self.args = args
        self.metric = args.trail_metric
        self.mode = args.trail_mode
        self.gpu_ids = args.gpu_ids
        self.configs = configs
        self.K = len(self.configs)

    def exploit(self):
        # gpu_ids = [0, 1, 0, 1]
        gpu_ids = [int(x) for x in self.gpu_ids.split(",")]
        n_gpu = len(gpu_ids)
        # 任务队列 & 结果队列
        task_queue = mp.Queue()
        result_queue = mp.Queue()
        # 启动每 GPU 一个 Worker 进程
        workers = []
        for gid in gpu_ids:
            w = GPUWorker(gid, self.args, task_queue, result_queue)
            w.start()
            workers.append(w)
        result = []
        current = 0  # 已经下发的任务数
        finished = 0  # 已经完成的任务数
        print(f"🔹 启动动态并发评估（每 GPU 1 个进程，共 {n_gpu} 个并行）")
        # 先给每个 GPU 发一个任务
        start_time = time.time()
        for i in range(min(self.K, 3*n_gpu)):
            cfg = copy.copy(self.configs[current])
            task_queue.put(cfg)
            current += 1
        print(f"🚀 下发新任务到 GPU （第 {current}/{self.K} 个）")

        # 主循环：不断拿结果 & 补充新任务
        while finished < self.K:
            cfg, metrics, gpu_id = result_queue.get()
            # print(gpu_id, metrics, cfg)
            result.append({
                "id": cfg["id"],
                "loss": metrics["loss"],
                "acc": metrics["acc"],
                "auc": metrics["auc"],
                "auc_history": metrics["auc_history"],
                "loss_history": metrics["loss_history"],
                "time": metrics["time"],
                "config": cfg,
            })
            finished += 1
            # 如果还有没下发的任务，继续下发
            if current < self.K:
                new_cfg = copy.copy(self.configs[current])
                task_queue.put(new_cfg)
                current += 1
                if current % 10 == 0:
                    spend_time = time.time() - start_time
                    start_time = time.time()
                    print(
                        f"🚀 下发新任务到 GPU （第 {current}/{self.K} 个）, "
                        f"Spend: {spend_time:.3f}"
                    )
        # 所有任务完成，向 worker 发送结束信号
        for _ in gpu_ids:
            task_queue.put(None)

        # 等待所有 GPU worker 退出
        for w in workers:
            w.join()

        # result.sort(key=lambda x: x[self.metric], reverse=True)
        print(f"全部任务完成，共 {len(result)} 个结果")
        return result

class BudgetAwareCoordinatorUniform:
    def __init__(self, args, budget: float, exploit_profile_time: float):
        self.budget = budget
        self.t2 = exploit_profile_time
        self.U_init = 1
        self.R = args.max_epochs
        self.num_workers = args.num_workers
        self.enable_at_least = 1 * self.R * self.U_init * self.t2

    def schedule(self):
        if self.budget < 1:
            raise Exception("budget must be larger than 1s")

        if self.budget < self.enable_at_least:
            C = 0
            T2_real = 0
            return C, self.budget, T2_real
        else:
            C = int((self.budget * self.num_workers) / (self.U_init * self.t2 * self.R) )
            T2_real = C * self.U_init * self.t2 * self.R / self.num_workers
            return C, self.budget, T2_real

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="frappe")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=40)
    parser.add_argument("--trail_metric", type=str, default="auc")
    parser.add_argument("--trail_mode", type=str, default="max")
    parser.add_argument("--exp_name", type=str, default="2phase-all")
    parser.add_argument("--reduction_factor", type=int, default=2)
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--swa_start_epoch", type=int, default=2)
    parser.add_argument("--budget", type=int, default=237)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device_ids", type=str, default="1")
    parser.add_argument("--gpu_ids", type=str, default="1")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print("========== Parsed Arguments ==========")
    for k, v in vars(args).items():
        print(f"{k:20s}: {v}")
    print("======================================")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_ids
    total_budget = args.budget
    kv = get_profile_data(dataset= args.dataset)
    t2 = kv["t2"]
    sh = BudgetAwareCoordinatorUniform(args=args, budget=total_budget, exploit_profile_time=t2)
    C, B, T2_real = sh.schedule()
    print(f"精选C: {C}, 预算B:{B}, 实际预算T2:{T2_real}")

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)

    configs = []
    randomSampler = RandomSearcher(search_space=reg_space, metric = args.trail_metric, mode = args.trail_mode, seed=args.seed)

    # 启动评估
    train_epochs = args.max_epochs   # 直接评估最后一个
    max_epochs = args.max_epochs
    for i in range(C):
        cfg = randomSampler.suggest()
        cfg["id"] = i
        cfg["train_epochs"] = train_epochs
        configs.append(cfg)

    print(f"len(configs): {len(configs)}")
    # print(f"configs: {configs}")



    exploitPhaseParallel = ExploitPhaseParallel(args=args, configs=configs)
    round_result = exploitPhaseParallel.exploit()
    # print(round_result)
    # 更新 configs 和 train_epoch
    res = {
        "Budget": total_budget,
        "T2_real": T2_real,
        "C": C,
        "round_result": numpy_to_python(round_result),
    }
    save_dict_to_file(data=res, base_dir=f"/data/ruipeng/workdir/autoreg/.exp_results/exp7/{args.dataset}/all", prefix=f"{args.exp_name}_{args.budget}")
    print("保存结果到文件")
    print("=======================================")
