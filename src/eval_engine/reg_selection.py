import argparse
import logging
import os
import time
import multiprocessing as mp
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
from src.data.datasets import get_dataset
from src.data.meta import get_metadata
from src.data.utils import compute_class_weights
from src.exp.util import set_seed, numpy_to_python, parse_results
from src.model.backbone import BackboneMLP
from src.trainer.step_trainer import StepTrainer
from src.space.space import reg_space
from src.searcher.area_searcher import AgeEvolutionSearcher
from src.trainer.trainer import Trainer


class ExplorePhaseSerial:
    def __init__(self, args):
        self.args = args
        # 评估器
        self.evaluator = ExploreEvaluator(args)
        self.N = args.num_samples
        self.population_size = args.population_size
        self.sample_size = args.sample_size
        self.metric = args.trail_metric
        self.mode = args.trail_mode
        self.sampler = AgeEvolutionSearcher(reg_space, self.population_size, self.sample_size, self.metric, self.mode)
        self.K = int(args.k_n * self.N)

    def explore(self, topK: bool = True):
        explore_current = 0
        # {"config": "", "loss": "", "acc": "", "bacc"}
        result = []
        while explore_current < self.N:
            cfg = self.sampler.suggest()
            metrics = self.evaluator.evaluate(cfg)
            self.sampler.on_result(cfg, metrics)
            explore_current += 1
            result.append({"config": cfg, "loss": metrics["loss"], "acc": metrics["acc"], "bacc": metrics["bacc"], "time": metrics["time"]})
            print(metrics)
        # 按照 bacc 降序排
        result.sort(key=lambda x: x[self.metric], reverse=True)
        return result[:self.K] if topK else result

def parallel_run_eval(args, cfg, gpu_id):
    args.device = f"cuda:{gpu_id}"
    evaluator = ExploreEvaluator(args)
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
        self.sampler = AgeEvolutionSearcher(reg_space, self.population_size, self.sample_size, self.metric, self.mode)
        self.K = int(args.k_n * self.N)

    def explore(self, topK: bool = True):
        gpu_ids = [0, 1, 2, 3]
        n_gpu = len(gpu_ids)
        pool = Pool(2*n_gpu)
        result = []
        running_tasks = []
        explore_current = 0
        print(f"🔹 启动动态并发评估（最多 2 * {n_gpu}(GPU) 并行）")
        # 先发前 n_gpu 个任务
        for i in range(min(self.N, 2*n_gpu)):
            cfg = self.sampler.suggest()
            gpu_id = gpu_ids[i % n_gpu]
            task = pool.apply_async(parallel_run_eval, (self.args, cfg, gpu_id))
            running_tasks.append(task)
            explore_current += 1
            print(f"🚀 下发新任务到 GPU {gpu_id} （第 {explore_current}/{self.N} 个）")

        # 动态监控任务完成情况
        while running_tasks:
            for task in running_tasks[:]:
                if task.ready():
                    cfg, metrics, gpu_id = task.get()
                    self.sampler.on_result(cfg, metrics)
                    print(metrics)
                    result.append({
                        "config": cfg,
                        "loss": metrics["loss"],
                        "acc": metrics["acc"],
                        "bacc": metrics["bacc"],
                        "time": metrics["time"]
                    })
                    # print(f"✅ GPU {gpu_id} 完成任务，结果：{metrics}")
                    running_tasks.remove(task)
                    # 如果还有没跑完的任务，就派发新的
                    if explore_current < self.N:
                        new_cfg = self.sampler.suggest()
                        new_task = pool.apply_async(parallel_run_eval, (self.args, new_cfg, gpu_id))
                        running_tasks.append(new_task)
                        explore_current += 1
                        print(f"🚀 下发新任务到 GPU {gpu_id} （第 {explore_current}/{self.N} 个）")

            time.sleep(0.05)  # 每隔 0.05 秒检查一次任务完成情况
        pool.close()
        pool.join()
        result.sort(key=lambda x: x[self.metric], reverse=True)
        print(f"全部任务完成，共 {len(result)} 个结果")
        return result[:self.K] if topK else result

class ExploreEvaluator:
    def __init__(self, args):
        self.args = args
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.device = args.device
        self.sample_ratio = args.sample_ratio
        self.max_steps = args.max_steps
        self.verbose = args.verbose
        self.meta = get_metadata(dataset=self.dataset)
        self.in_features = self.meta["in_features"]
        self.out_features = self.meta["out_features"]
        self.hidden_features = [512, 512, 512, 512, 512, 512]
        self.is_balanced = self.meta["is_balanced"]
        self.class_ratio = self.meta["class_ratio"]
        self.data_dir = self.meta["data_dir"]
        self.seed = args.seed
        self.train_loader, self.valid_loader, self.test_loader = get_sampled_dataloader(
            dataset=self.dataset, batch_size=self.batch_size,
            data_dir=self.data_dir, sample_ratio=self.sample_ratio,
        )
        set_seed(self.seed)
        self.weights = None
        if not self.is_balanced:
            self.weights = compute_class_weights(self.class_ratio, method="inv")
        if self.weights is not None:
            ce_weight = torch.tensor(self.weights, dtype=torch.float32).to(torch.device(self.device))
        else:
            ce_weight = None
        self.criterion = nn.CrossEntropyLoss(weight=ce_weight)

    def evaluate(self, config) -> dict[str, Any]:
        start_time = time.time()
        # 初始化模型
        model = BackboneMLP(
            input_dim=self.in_features,
            hidden_dims=self.hidden_features,
            output_dim=self.out_features,
            reg_config=config,
        )
        # 初始化训练器
        trainer = StepTrainer(
            model=model,
            criterion=self.criterion,
            optimizer_name="AdamW",
            lr=self.args.lr,
            momentum=self.args.momentum,
            device=self.device,
            reg_config=config,
        )
        trainer.train(self.train_loader, self.valid_loader, max_steps=self.max_steps, val_interval=self.max_steps, verbose=self.verbose)
        loss, acc, bacc = trainer.evaluate(self.test_loader)
        metrics = {
            "loss": loss,
            "acc": acc,
            "bacc": bacc,
            "time": time.time() - start_time,
        }
        return metrics


def exploitation_train(config, args, train_set, val_set, test_set):
    print("Visible GPUs:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    config = config["config"]
    set_seed(args.seed)
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
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
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
        optimizer_name="AdamW",
        lr=args.lr,
        momentum=args.momentum,
        swa_start_epoch=swa_start_epoch,
        device=device,
        reg_config=config,
    )
    for epoch in range(max_epochs):
        trainer.train(train_loader, valid_loader, epochs=1, verbose=args.verbose)
        loss, acc, bacc = trainer.evaluate(test_loader)
        metrics = {
            "loss": loss,
            "acc": acc,
            "bacc": bacc,
        }
        tune.report(metrics)


class ExploitPhase:
    def __init__(self, args):
        self.args = args
        self.dataset = args.dataset
        self.meta = get_metadata(dataset=self.dataset)
        self.data_dir = self.meta["data_dir"]
        self.train_set, self.val_set, self.test_set = get_dataset(self.dataset, self.data_dir)

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
            )
        )
        results = tuner.fit()
        df = results.get_dataframe()
        return parse_results(df)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="adult")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_cpus", type=int, default=8)
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--max_concurrent_trials", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--max_epochs", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=50)
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
    parser.add_argument("--swa_start_epoch", type=int, default=2)
    # sample_ratio
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    explorePhase = ExplorePhaseParallel(args)
    start_time = time.time()
    result = explorePhase.explore(topK=True)
    end_time = time.time()
    print(f"总时间：{end_time - start_time}")
    # print("结果")
    # print(result)
    init_time = time.time()
    ray.init(num_cpus=args.num_cpus, num_gpus=args.num_gpus, include_dashboard=False, configure_logging=False,
             logging_level=logging.ERROR)
    rs = ray.available_resources()
    print(f"集群可用资源：\n{rs}")
    print(f"初始化集群时间：{time.time() - init_time}")

    start_time2 = time.time()
    configs = [item["config"] for item in result]
    configs = numpy_to_python(configs)
    print(configs)
    exploitPhase = ExploitPhase(args)
    res2 = exploitPhase.exploit(configs)
    # print(configs)
    print(f"利用时间:{time.time() - start_time2}")
    print(res2)