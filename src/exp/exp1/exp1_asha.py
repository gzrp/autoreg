import argparse
import logging
import os
import random
import time

import numpy as np
import ray
import torch
import torch.nn as nn
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import Tuner, TuneConfig, RunConfig
from torch.utils.data import DataLoader

from src.data.datasets import get_dataset
from src.data.meta import get_metadata
from src.data.utils import compute_class_weights
from src.space.space import reg_space
from src.exp.exp1.util import set_seed, parse_results
from src.model.backbone import BackboneMLP
from src.trainer.trainer import Trainer
from src.utils.util import numpy_to_python, save_dict_to_file


def asha_train(config, args, train_set, val_set, test_set):
    print("Visible GPUs:", os.environ.get("CUDA_VISIBLE_DEVICES"))
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
    for epoch in range(max_epochs):
        trainer.train(train_loader, valid_loader, epochs=1, verbose=args.verbose)
        loss, acc, bacc = trainer.evaluate(test_loader)
        metrics = {
            "loss": loss,
            "acc": acc,
            "bacc": bacc,
        }
        tune.report(metrics)

def asha_phase(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # 配置数据，一次加载
    dataset = args.dataset
    meta = get_metadata(dataset=dataset)
    data_dir = meta["data_dir"]
    start_time = time.time()
    train_set, val_set, test_set = get_dataset(dataset, data_dir)
    print(f"加载 dataset {time.time() - start_time}")
    # asha 算法
    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        metric=args.trail_metric,
        mode=args.trail_mode,
        max_t=args.max_epochs,
        grace_period=1,
        reduction_factor=args.reduction_factor,
    )
    tuner = Tuner(
        trainable=tune.with_resources(
            tune.with_parameters(asha_train, args=args, train_set=train_set, val_set=val_set, test_set=test_set),
            resources={"cpu": args.trail_num_cpus, "gpu": args.trail_num_gpus}
        ),
        param_space=reg_space,
        tune_config=TuneConfig(
            num_samples=args.num_samples,
            scheduler=scheduler,
        ),
        run_config=RunConfig(
            name=args.exp_name,
            storage_path=args.storage,
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
    parser.add_argument("--num_cpus", type=int, default=10)
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--trail_num_cpus", type=int, default=2)
    parser.add_argument("--trail_num_gpus", type=float, default=1)
    parser.add_argument("--trail_metric", type=str, default="bacc")
    parser.add_argument("--trail_mode", type=str, default="max")
    parser.add_argument("--exp_name", type=str, default="asha")
    parser.add_argument("--storage", type=str, default="~/ray_results")
    parser.add_argument("--reduction_factor", type=int, default=2)
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--swa_start_epoch", type=int, default=1)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    init_time = time.time()
    ray.init(num_cpus=args.num_cpus, num_gpus=args.num_gpus, include_dashboard=False, configure_logging=False, logging_level=logging.ERROR)
    rs = ray.available_resources()
    print("---" * 100)
    print(f"集群可用资源：{rs}")
    print(f"初始化集群时间：{time.time() - init_time} s")
    print("---" * 100)

    start_time = time.time()
    res = asha_phase(args)
    total_time = time.time() - start_time
    save_res = {"total_time": total_time, "items": res}
    print(f"总时间：{total_time} s")
    print(f"最佳配置：{res[0]}")
    res = numpy_to_python(res)
    save_result = {
        "total_time": total_time,
        "asha_num": len(res),
        "best": res[0],
        "asha": res
    }
    save_dict_to_file(data=save_result, base_dir="/data/ruipeng/workdir/autoreg/.exp_results", prefix=args.exp_name)
