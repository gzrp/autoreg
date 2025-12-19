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

from src.data.dataloaders import get_dataloader
from src.data.datasets import get_dataset_sampled
from src.data.meta import get_metadata
from src.data.utils import compute_class_weights
from src.model.backbone import BackboneMLP
from src.profiling.profiling import get_profile_data
from src.trainer.step_trainer import StepTrainer
from src.space.space import reg_space
from src.searcher.area_searcher import AgeEvolutionSearcher
from src.trainer.trainer_new import Trainer
from src.utils.util import numpy_to_python, append_jsonl
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
    train_set, valid_set, test_set = get_dataset_sampled(
        dataset=args.dataset,
        data_dir=data_dir,
        sample_ratio=args.sample_ratio
    )
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
            torch.cuda.set_device(0)
            torch.cuda.manual_seed(seed)

    def run(self):
        # 在当前进程里初始化 evaluator，只做一次
        evaluator = ExploreEvaluator(self.args)

        while True:
            cfg = self.task_queue.get()
            if cfg is None:
                # 收到结束信号
                break
            metrics = evaluator.evaluate(cfg)
            # 把结果放回主进程
            self.result_queue.put((cfg, metrics, self.gpu_id))

class ExplorePhaseParallel:
    def __init__(self, args):
        self.args = args
        self.N = args.num_samples
        self.population_size = args.population_size
        self.sample_size = args.sample_size
        self.metric = args.trail_metric
        self.mode = args.trail_mode
        self.sampler = AgeEvolutionSearcher(reg_space, self.population_size, self.sample_size, self.metric, self.mode, args.seed)

        self.K = int(args.k_n * self.N)

    def explore(self, topK: bool = True):
        gpu_ids = [0, 1, 0, 1]
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
        explore_current = 0  # 已经下发的任务数
        finished = 0  # 已经完成的任务数

        print(f"🔹 启动动态并发评估（每 GPU 1 个进程，共 {n_gpu} 个并行）")

        # 先给每个 GPU 发一个任务
        start_time = time.time()
        for i in range(min(self.N, 3*n_gpu)):
            cfg = self.sampler.suggest()
            task_queue.put(cfg)
            explore_current += 1
        print(f"🚀 下发新任务到 GPU （第 {explore_current}/{self.N} 个）")

        # 主循环：不断拿结果 & 补充新任务
        while finished < self.N:
            cfg, metrics, gpu_id = result_queue.get()
            self.sampler.on_result(cfg, metrics)
            print(gpu_id, metrics, cfg)
            result.append({
                "loss": metrics["loss"],
                "acc": metrics["acc"],
                "auc": metrics["auc"],
                "time": metrics["time"],
                "config": cfg,
            })
            finished += 1

            # 如果还有没下发的任务，继续下发
            if explore_current < self.N:
                sample_start_time = time.time()
                new_cfg = self.sampler.suggest()
                sample_time = time.time() - sample_start_time
                task_queue.put(new_cfg)
                explore_current += 1
                if explore_current % 50 == 0:
                    spend_time = time.time() - start_time
                    start_time = time.time()
                    print(
                        f"🚀 下发新任务到 GPU （第 {explore_current}/{self.N} 个）, "
                        f"Spend: {spend_time:.3f}, Sample: {sample_time:.6f}"
                    )
        # 所有任务完成，向 worker 发送结束信号
        for _ in gpu_ids:
            task_queue.put(None)

        # 等待所有 GPU worker 退出
        for w in workers:
            w.join()

        result.sort(key=lambda x: x[self.metric], reverse=True)
        print(f"全部任务完成，共 {len(result)} 个结果")
        return result, result[:self.K] if topK else result

class ExploreEvaluator:
    def __init__(self, args):
        self.args = args
        self.seed = args.seed
        self.device = args.device

        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.swa_start_epoch = args.swa_start_epoch

        self.sample_ratio = args.sample_ratio
        self.max_steps = args.max_steps
        self.verbose = args.verbose

        init_global_dataset(args)
        # ===== 使用全局 META/DATASET，而不是每次重新 get_* =====
        global GLOBAL_DATASET_META, GLOBAL_TRAIN_SET, GLOBAL_VALID_SET, GLOBAL_TEST_SET

        if GLOBAL_DATASET_META is not None:
            self.meta = GLOBAL_DATASET_META
        else:
            self.meta = get_metadata(dataset=self.dataset)

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
        trainer = StepTrainer(
            model=model,
            criterion=criterion,
            lr=self.args.lr,
            device=self.device,
            reg_config=config,
            metric_type="AUC",
        )
        trainer.train(train_loader, max_steps=self.max_steps)

        loss, acc, auc = trainer.evaluate(test_loader)

        metrics = {
            "loss": loss,
            "acc": acc,
            "auc": auc,
            "time": time.time() - start_time,
        }
        # -------------------------------------------------
        # ⭐⭐ 强烈推荐：在这里清理模型和显存 ⭐⭐
        # -------------------------------------------------
        torch.cuda.synchronize()  # 让 CUDA 异步执行完
        del model
        del trainer
        del criterion
        torch.cuda.empty_cache()
        return metrics

class BudgetAwareCoordinatorSH:
    def __init__(self, args, budget: float, explore_profile_time: float, exploit_profile_time: float, only_one_phase: bool = False):
        self.budget = budget
        self.eta = args.reduction_factor
        self.t1 = explore_profile_time
        self.t2 = exploit_profile_time
        self.alpha = args.k_n
        self.U_init = 1
        self.R = args.max_epochs
        self.num_workers = args.num_workers
        self.only_one_phase = only_one_phase

    def schedule(self):
        if self.budget < 1:
            raise Exception("budget must be larger than 1s")

        enable_phase2_at_least = self.t1 / self.alpha / self.num_workers + self.R * self.U_init * self.t2
        if self.only_one_phase or self.budget < enable_phase2_at_least:
            N = int(self.budget / self.t1 * self.num_workers)
            C = 0
            T1_real = N * self.t1 / self.num_workers
            T2_real = 0
            T_real = T1_real + T2_real
            return N, C, self.budget, T_real, T1_real, T2_real
        else:
            k = int(math.log(self.R, self.eta))
            C = int((self.budget * self.num_workers) / (self.t1 / self.alpha + self.U_init * self.t2 * (k+1)) )
            N = int(C / self.alpha)
            T1_real = N * self.t1 / self.num_workers
            T2_real = C * self.U_init * self.t2 * (k+1) / self.num_workers
            T_real = T1_real + T2_real
            return N, C, self.budget, T_real, T1_real, T2_real
        # print("enable_phase2_at_least", enable_phase2_at_least)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="bank")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_cpus", type=int, default=10)
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=40)
    parser.add_argument("--trail_num_cpus", type=int, default=2)
    parser.add_argument("--trail_num_gpus", type=float, default=0.5)
    parser.add_argument("--trail_metric", type=str, default="auc")
    parser.add_argument("--trail_mode", type=str, default="max")
    parser.add_argument("--exp_name", type=str, default="2phase")
    parser.add_argument("--storage", type=str, default="~/ray_results")
    parser.add_argument("--reduction_factor", type=int, default=2)
    parser.add_argument("--population_size", type=int, default=10)
    parser.add_argument("--sample_size", type=int, default=3)
    parser.add_argument("--k_n", type=float, default=0.2)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--sample_ratio", type=float, default=0.2)
    parser.add_argument("--swa_start_epoch", type=int, default=2)
    parser.add_argument("--budget", type=int, default=21)
    parser.add_argument("--grace_period", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device_ids", type=str, default="0,1")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_ids
    total_budget = args.budget
    kv = get_profile_data(dataset= args.dataset)
    t1 = kv["t1"]
    t2 = kv["t2"]
    sh = BudgetAwareCoordinatorSH(args=args, budget=total_budget, explore_profile_time=t1, exploit_profile_time=t2, only_one_phase=True)
    N, C, B_real, T_real, T1_real, T2_real = sh.schedule()
    print(f"N: {N}, C:{C}")
    args.num_samples = N

    print("========== Parsed Arguments ==========")
    for k, v in vars(args).items():
        print(f"{k:20s}: {v}")
    print("======================================")

    res = {
        "Budget": total_budget,
        "T_real": T_real,
        "T1_real": T1_real,
        "T2_real": T2_real,
        "N": N,
        "C": C,
        "train_result": None,
        "best_explore": None,
    }
    print(f"探索{N}, 精选{C}")
    print("======================================")
    explore_start_time = time.time()
    explorePhase = ExplorePhaseParallel(args)
    all_res, res1 = explorePhase.explore(topK=True)
    explore_time = time.time() - explore_start_time
    best = all_res[0]
    best = numpy_to_python(best)
    print(f"探索时间：{explore_time} s")
    print(f"探索最佳配置：{best}")
    res["best_explore"] = best
    config = best["config"]
    print(config)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    # 数据准备
    meta = get_metadata(dataset=args.dataset)
    in_features = meta["in_features"]
    out_features = meta["out_features"]
    is_balanced = meta["is_balanced"]
    class_ratio = meta["class_ratio"]
    data_dir = meta["data_dir"]
    batch_size = args.batch_size


    train_loader, valid_loader, test_loader = get_dataloader(dataset=args.dataset ,data_dir=data_dir, batch_size=batch_size)

    # 初始化模型
    hidden_features = [512, 512, 512, 512, 512, 512]
    # config = {}
    model = BackboneMLP(
        input_dim=in_features,
        hidden_dims=hidden_features,
        output_dim=out_features,
        reg_config=config,
    )
    # 初始化训练器
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ce_weight = None
    if not is_balanced:
        weights = compute_class_weights(class_ratio, method="inv")
        ce_weight = torch.tensor(weights, dtype=torch.float32).to(torch.device(device))

    criterion = nn.CrossEntropyLoss(weight=ce_weight)
    trainer = Trainer(
        model=model,
        criterion=criterion,
        lr=1e-3,
        device=device,
        reg_config=config,
        metric_type="AUC",
    )
    loss = 0
    acc = 0
    auc = 0
    loss_history = []
    auc_history = []
    for epoch in range(args.max_epochs):
        trainer.train(train_loader, valid_loader, epochs=1, verbose=True)
        loss, acc, auc = trainer.evaluate(test_loader)
        auc_history.append(auc)
        loss_history.append(loss)
    res["train_result"] = {
        "loss": loss,
        "acc": acc,
        "auc": auc,
        "auc_history": auc_history,
        "loss_history": loss_history,
    }
    append_jsonl(res, f"/data/ruipeng/workdir/autoreg/.exp_results/auc/logs/{args.dataset}/1pahse/1phase_time_log.jsonl")
    print("保存结果到文件")
    print("=======================================")


