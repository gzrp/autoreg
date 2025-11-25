import argparse
import copy
import logging
import math
import os
import random
import time

import numpy as np
import ray
import torch
import torch.nn as nn
from typing import Any
from multiprocessing import Pool
from ray import tune
from ray.tune import Tuner, TuneConfig, RunConfig
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import DataLoader

from src.data.dataloaders import get_sampled_dataloader
from src.data.datasets import get_dataset_sampled
from src.data.datasets import get_dataset
from src.data.meta import get_metadata
from src.data.utils import compute_class_weights
from src.exp.exp1.util import parse_results
from src.model.backbone import BackboneMLP
from src.profiling.profiling import get_profile_data
from src.trainer.step_trainer import StepTrainer
from src.space.space import reg_space, get_default_reg
from src.searcher.area_searcher import AgeEvolutionSearcher
from src.trainer.trainer import Trainer
from src.utils.util import numpy_to_python, append_jsonl


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

def parallel_run_eval(args, cfg, gpu_id):
    local_args = copy.copy(args)
    local_args.device = f"cuda:{gpu_id}"
    # 在子进程里构造 evaluator（里面会做 CUDA & Dataset 初始化）
    evaluator = ExploreEvaluator(local_args)
    metrics = evaluator.evaluate(cfg)
    return cfg, metrics, gpu_id


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

    def profiling(self):
        start_time = time.time()
        local_args = copy.copy(args)
        local_args.device = f"cuda:{0}"
        evaluator = ExploreEvaluator(local_args)
        cfg = get_default_reg()
        metrics = evaluator.evaluate(cfg)
        total_time = time.time() - start_time
        return total_time

    def explore(self, topK: bool = True):
        # 创建进程池前，主进程加载一次数据集
        # init_global_dataset(self.args)
        # gpu_ids = [0, 1, 2, 3]
        gpu_ids = [2, 3]
        n_gpu = len(gpu_ids)
        pool = Pool(2*n_gpu)
        result = []
        running_tasks = []
        explore_current = 0
        print(f"🔹 启动动态并发评估（最多 2 * {n_gpu}(GPU) 并行）")
        # 先发前 n_gpu 个任务
        start_time = time.time()
        for i in range(min(self.N, 2*n_gpu)):
            cfg = self.sampler.suggest()
            gpu_id = gpu_ids[i % n_gpu]
            task = pool.apply_async(parallel_run_eval, (self.args, cfg, gpu_id))
            running_tasks.append(task)
            explore_current += 1

        print(f"🚀 下发新任务到 GPU （第 {explore_current}/{self.N} 个）")
        # 动态监控任务完成情况
        while running_tasks:
            for task in running_tasks[:]:
                if task.ready():
                    cfg, metrics, gpu_id = task.get()
                    self.sampler.on_result(cfg, metrics)
                    print(metrics, cfg)
                    result.append({
                        "loss": metrics["loss"],
                        "acc": metrics["acc"],
                        "bacc": metrics["bacc"],
                        "time": metrics["time"],
                        "config": cfg,
                    })
                    # print(f"✅ GPU {gpu_id} 完成任务，结果：{metrics}")
                    running_tasks.remove(task)
                    # 如果还有没跑完的任务，就派发新的
                    if explore_current < self.N:
                        sample_start_time = time.time()
                        new_cfg = self.sampler.suggest()
                        sample_time = time.time() - sample_start_time
                        new_task = pool.apply_async(parallel_run_eval, (self.args, new_cfg, gpu_id))
                        running_tasks.append(new_task)
                        explore_current += 1
                        if explore_current % 50 == 0:
                            spend_time = time.time() - start_time
                            start_time = time.time()
                            print(f"🚀 下发新任务到 GPU （第 {explore_current}/{self.N} 个）, Spend: {spend_time}, Sample: {sample_time}")

            time.sleep(0.05)  # 每隔 0.05 秒检查一次任务完成情况
        pool.close()
        pool.join()
        result.sort(key=lambda x: x[self.metric], reverse=True)
        print(f"全部任务完成，共 {len(result)} 个结果")
        return result, result[:self.K] if topK else result

class ExploreEvaluator:
    def __init__(self, args):
        self.args = args
        self.seed = args.seed
        self.device = args.device
        # set_seed(self.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.set_device(args.device)
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
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

        # self.meta = get_metadata(dataset=self.dataset)
        self.in_features = self.meta["in_features"]
        self.out_features = self.meta["out_features"]
        self.hidden_features = [512, 512, 512, 512, 512, 512]
        self.is_balanced = self.meta["is_balanced"]
        self.class_ratio = self.meta["class_ratio"]
        self.data_dir = self.meta["data_dir"]

        # ⭐ 如果全局 DATASET 已初始化，则在当前进程中构建 DataLoader
        if GLOBAL_TRAIN_SET is not None and GLOBAL_VALID_SET is not None and GLOBAL_TEST_SET is not None:
            train_set = GLOBAL_TRAIN_SET
            valid_set = GLOBAL_VALID_SET
            test_set = GLOBAL_TEST_SET

            # 这里每个进程自己建 DataLoader，开销很小
            # 是否 shuffle 可以按你需求调整，这里给出一种保守写法
            self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=False)
            self.valid_loader = DataLoader(valid_set, batch_size=self.batch_size, shuffle=False)
            self.test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)
        else:
            # 兜底逻辑：如果没初始化全局数据，保持原始行为
            self.train_loader, self.valid_loader, self.test_loader = get_sampled_dataloader(
                dataset=self.dataset, batch_size=self.batch_size,
                data_dir=self.data_dir, sample_ratio=self.sample_ratio,
            )

    def profile_filtering(self):
        begin_time = time.time()
        # 初始化模型
        model = BackboneMLP(
            input_dim=self.in_features,
            hidden_dims=self.hidden_features,
            output_dim=self.out_features,
            reg_config=None,
        )
        weights = None
        if not self.is_balanced:
            weights = compute_class_weights(self.class_ratio, method="inv")
        if weights is not None:
            ce_weight = torch.tensor(weights, dtype=torch.float32).to(torch.device(self.device))
        else:
            ce_weight = None
        criterion = nn.CrossEntropyLoss(weight=ce_weight)
        # 初始化训练器
        trainer = StepTrainer(
            model=model,
            criterion=criterion,
            lr=self.args.lr,
            device=self.device,
            reg_config=None,
        )
        trainer.train(self.train_loader, max_steps=self.max_steps)
        loss, acc, bacc = trainer.evaluate(self.test_loader)
        score_time_per_reg = time.time() - begin_time
        # metrics = {
        #     "loss": loss,
        #     "acc": acc,
        #     "bacc": bacc,
        #     "time": score_time_per_reg,
        # }
        # if self.verbose:
        #     print(metrics)
        return score_time_per_reg

    def evaluate(self, config) -> dict[str, Any]:
        start_time = time.time()
        # 初始化模型
        model = BackboneMLP(
            input_dim=self.in_features,
            hidden_dims=self.hidden_features,
            output_dim=self.out_features,
            reg_config=config,
        )
        weights = None
        if not self.is_balanced:
            weights = compute_class_weights(self.class_ratio, method="inv")
        if weights is not None:
            ce_weight = torch.tensor(weights, dtype=torch.float32).to(torch.device(self.device))
        else:
            ce_weight = None
        criterion = nn.CrossEntropyLoss(weight=ce_weight)
        # 初始化训练器
        trainer = StepTrainer(
            model=model,
            criterion=criterion,
            lr=self.args.lr,
            device=self.device,
            reg_config=config,
        )
        trainer.train(self.train_loader, max_steps=self.max_steps)

        loss, acc, bacc = trainer.evaluate(self.test_loader)

        metrics = {
            "loss": loss,
            "acc": acc,
            "bacc": bacc,
            "time": time.time() - start_time,
        }
        return metrics



def exploitation_profiling(args, train_set, val_set, test_set):
    # print("Visible GPUs:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    # config = config["config"]
    # set_seed(args.seed)
    config = get_default_reg()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    # 数据准备
    dataset = args.dataset
    batch_size = args.batch_size
    device = args.device
    swa_start_epoch = args.swa_start_epoch
    max_epochs = args.max_epochs
    meta = get_metadata(dataset=dataset)
    in_features = meta["in_features"]
    out_features = meta["out_features"]
    is_balanced = meta["is_balanced"]
    class_ratio = meta["class_ratio"]
    weights = None
    if not is_balanced:
        weights = compute_class_weights(class_ratio, method="inv")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    # 初始化模型
    hidden_features = [512, 512, 512, 512, 512, 512]
    model = BackboneMLP(
        input_dim=in_features,
        hidden_dims=hidden_features,
        output_dim=out_features,
        reg_config=config,
    )
    # 初始化训练器
    if weights is not None:
        ce_weight = torch.tensor(weights, dtype=torch.float32).to(torch.device(device))
    else:
        ce_weight = None

    criterion = nn.CrossEntropyLoss(weight=ce_weight)
    trainer = Trainer(
        model=model,
        criterion=criterion,
        lr=args.lr,
        swa_start_epoch=swa_start_epoch,
        device=device,
        reg_config=config,
    )
    acc_max = 0
    bacc_max = 0
    for epoch in range(max_epochs):
        t1 = time.time()
        trainer.train(train_loader, valid_loader, epochs=1, verbose=args.verbose)
        loss, acc, bacc = trainer.evaluate(test_loader)
        acc_max = max(acc_max, acc)
        bacc_max = max(bacc_max, bacc)
        metrics = {
            "loss": loss,
            "acc": acc_max,
            "bacc": bacc_max,
        }
        # tune.report(metrics)
        t2 = time.time()
        print(f"{epoch} Time: {t2-t1}")



def exploitation_train(config, args, train_set, val_set, test_set):
    print("Visible GPUs:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    config = config["config"]
    # set_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    # 数据准备
    dataset = args.dataset
    batch_size = args.batch_size
    device = args.device
    swa_start_epoch = args.swa_start_epoch
    max_epochs = args.max_epochs
    meta = get_metadata(dataset=dataset)
    in_features = meta["in_features"]
    out_features = meta["out_features"]
    is_balanced = meta["is_balanced"]
    class_ratio = meta["class_ratio"]
    weights = None
    if not is_balanced:
        weights = compute_class_weights(class_ratio, method="inv")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    # 初始化模型
    hidden_features = [512, 512, 512, 512, 512, 512]
    model = BackboneMLP(
        input_dim=in_features,
        hidden_dims=hidden_features,
        output_dim=out_features,
        reg_config=config,
    )
    # 初始化训练器
    if weights is not None:
        ce_weight = torch.tensor(weights, dtype=torch.float32).to(torch.device(device))
    else:
        ce_weight = None

    criterion = nn.CrossEntropyLoss(weight=ce_weight)
    trainer = Trainer(
        model=model,
        criterion=criterion,
        lr=args.lr,
        swa_start_epoch=swa_start_epoch,
        device=device,
        reg_config=config,
    )
    acc_max = 0
    bacc_max = 0
    val_bacc_max = []
    for epoch in range(max_epochs):
        trainer.train(train_loader, valid_loader, epochs=1, verbose=args.verbose)
        loss, acc, bacc = trainer.evaluate(test_loader)

        acc_max = max(acc_max, acc)
        if bacc > bacc_max:
            bacc_max = bacc
            val_bacc_max = trainer.val_bacc_history

        metrics = {
            "loss": loss,
            "acc": acc_max,
            "bacc": bacc_max,
            "bacc_history": val_bacc_max,
            "bacc_current": trainer.val_bacc_history
        }
        tune.report(metrics)


class ExploitPhase:
    def __init__(self, args):
        self.args = args
        self.dataset = args.dataset
        self.meta = get_metadata(dataset=self.dataset)
        self.data_dir = self.meta["data_dir"]
        self.train_set, self.val_set, self.test_set = get_dataset(self.dataset, self.data_dir)


    def profiling(self, max_epochs):
        start_time = time.time()
        local_args = copy.copy(args)
        local_args.device = f"cuda:{0}"
        local_args.max_epochs = max_epochs

        exploitation_profiling(local_args, self.train_set, self.val_set, self.test_set)
        total_time = time.time() - start_time
        return total_time

    def exploit(self, configs):
        scheduler = ASHAScheduler(
            time_attr="training_iteration",
            metric=self.args.trail_metric,
            mode=self.args.trail_mode,
            max_t=self.args.max_epochs,
            grace_period=1,
            reduction_factor=self.args.reduction_factor,
        )
        tuner = Tuner(
            trainable=tune.with_resources(
                tune.with_parameters(exploitation_train, args=self.args, train_set=self.train_set, val_set=self.val_set,
                                     test_set=self.test_set),
                resources={"cpu": self.args.trail_num_cpus, "gpu": self.args.trail_num_gpus}
            ),
            # grid_search, num_sample = 1
            tune_config=TuneConfig(
                scheduler=scheduler,
                num_samples=1
            ),
            param_space={"config": tune.grid_search(configs)},
            run_config=RunConfig(
                name=self.args.exp_name + "_exploitationPhase",
                storage_path=self.args.storage,
            ),

        )
        results = tuner.fit()
        df = results.get_dataframe()
        return parse_results(df)


class BudgetAwareCoordinatorSH:
    def __init__(self, args, budget: float, explore_profile_time: float, exploit_profile_time: float, only_one_phase: bool = False):
        self.budget = budget
        self.eta = args.reduction_factor
        self.t1 = explore_profile_time
        self.t2 = exploit_profile_time
        self.alpha = args.k_n
        self.U_init = 1
        self.R = args.max_epochs
        self.num_workers = args.num_gpus
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
    parser.add_argument("--dataset", type=str, default="connect")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_cpus", type=int, default=10)
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--trail_num_cpus", type=int, default=2)
    parser.add_argument("--trail_num_gpus", type=float, default=1)
    parser.add_argument("--trail_metric", type=str, default="bacc")
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
    parser.add_argument("--swa_start_epoch", type=int, default=1)
    parser.add_argument("--budget", type=int, default=48)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    init_time = time.time()
    ray.init(num_cpus=args.num_cpus, num_gpus=args.num_gpus, include_dashboard=False, configure_logging=False,
             logging_level=logging.ERROR)
    rs = ray.available_resources()
    print("---" * 100)
    print(f"集群可用资源：{rs}")
    print(f"初始化集群时间：{time.time() - init_time} s")
    print("---" * 100)

    total_budget = args.budget
    kv = get_profile_data(dataset= args.dataset)
    t1 = kv["t1"]
    t2 = kv["t2"]
    sh = BudgetAwareCoordinatorSH(args=args, budget=total_budget, explore_profile_time=t1, exploit_profile_time=t2)
    N, C, B_real, T_real, T1_real, T2_real = sh.schedule()

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
        "best_explore": None,
        "best_exploit": None,
    }
    print(f"探索{N}, 精选{C}")
    print("======================================")
    explore_start_time = time.time()
    explorePhase = ExplorePhaseParallel(args)
    all_res, res1 = explorePhase.explore(topK=True)
    explore_time = time.time() - explore_start_time
    print(f"探索时间：{explore_time} s")
    print(f"探索最佳配置：{all_res[0]}")
    res["best_explore"] = all_res[0]

    print("======================================")

    if C>0:
        exploit_start_time = time.time()
        configs = [item["config"] for item in res1]
        configs = numpy_to_python(configs)
        exploitPhase = ExploitPhase(args)
        res2 = exploitPhase.exploit(configs)
        exploit_time = time.time() - exploit_start_time
        res["best_exploit"] = res2[0]
        print(f"利用时间:{exploit_time} s")
        print("---" * 100)
        print(f"总时间：{explore_time + exploit_time} s")
        print(f"最佳配置：{res2[0]}")

    print("=======================================")
    print(res)

    append_jsonl(res, f"/data/ruipeng/workdir/autoreg/.exp_results/logs/{args.dataset}/2phase/2phase_time_log.jsonl")
    print("保存结果到文件")
    print("=======================================")