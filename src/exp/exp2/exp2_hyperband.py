import argparse
import json
import logging
import os
import random
import tempfile
import time

import numpy as np
import ray
import torch
import torch.nn as nn
from ray import tune
from ray.tune.schedulers import HyperBandScheduler
from ray.tune import Tuner, TuneConfig, RunConfig
from torch.utils.data import DataLoader

from src.data.datasets import get_dataset
from src.data.meta import get_metadata
from src.data.utils import compute_class_weights
from src.exp.exp2.util import BufferedBestSampler
from src.space.space import reg_space
from src.exp.exp1.util import set_seed, parse_results
from src.model.backbone import BackboneMLP
from src.trainer.trainer import Trainer
from src.utils.util import numpy_to_python, save_dict_to_file



def hyperband_train(config, args, train_set, val_set, test_set):
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
    checkpoint = tune.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
           trainer.load_model(os.path.join(checkpoint_dir, "checkpoint.pt"))

    val_bacc_history = []
    for epoch in range(max_epochs):
        trainer.train(train_loader, valid_loader, epochs=1, verbose=args.verbose)
        loss, acc, bacc = trainer.evaluate(test_loader)
        val_bacc_history.append(bacc)
        metrics = {
            "loss": loss,
            "acc": acc,
            "bacc": bacc,
            "bacc_history": val_bacc_history
        }
        # tune.report(metrics)
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            trainer.save_model(path)
            checkpoint = tune.Checkpoint.from_directory(temp_checkpoint_dir)
            tune.report(metrics, checkpoint=checkpoint)

def hyperband_phase(args):
    start_time = time.time()
    # 配置数据，一次加载
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    dataset = args.dataset
    meta = get_metadata(dataset=dataset)
    data_dir = meta["data_dir"]

    train_set, val_set, test_set = get_dataset(dataset, data_dir)
    print(f"加载 dataset {time.time() - start_time}")
    # hyperband 算法
    scheduler = HyperBandScheduler(
        time_attr="training_iteration",
        metric=args.trail_metric,
        mode=args.trail_mode,
        max_t=args.max_epochs,
        reduction_factor = args.reduction_factor,
    )
    callback = BufferedBestSampler(
        exp_name="hyperband",
        dataset=args.dataset,
        metric="bacc",
        mode="max",
        max_epochs=args.max_epochs,
        start_time=start_time,
        log_file="hyperband_time_log.jsonl",
        flush_every=10,
        verbose=True
    )
    tuner = Tuner(
        trainable=tune.with_resources(
            tune.with_parameters(hyperband_train, args=args, train_set=train_set, val_set=val_set, test_set=test_set),
            resources={"cpu": args.trail_num_cpus, "gpu": args.trail_num_gpus}
        ),
        param_space = reg_space,
        tune_config=TuneConfig(
            num_samples=args.num_samples,
            scheduler=scheduler,
        ),
        run_config=RunConfig(
            name=args.exp_name,
            storage_path=args.storage,
            callbacks=[callback],
            verbose=1
        )
    )
    results = tuner.fit()
    df = results.get_dataframe()
    return parse_results(df)

def get_result():
    tuner = Tuner.restore("~/ray_results/hyperband", trainable=hyperband_train)
    results = tuner.get_results()
    print(len(results))
    print(results)
    df = results.get_dataframe()
    res = parse_results(df)
    print(res)
    total_time = 5412.24
    save_res = {"total_time": total_time, "items": res}
    print(f"总时间：{total_time} s")
    print(f"最佳配置：{res[0]}")

    res = numpy_to_python(res)
    save_result = {
        "total_time": total_time,
        "hyperband_num": len(res),
        "best": res[0],
        "hyperband": res
    }
    save_dict_to_file(data=save_result, base_dir="/data/ruipeng/workdir/autoreg/.exp_results", prefix="hyperband")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dionis")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_cpus", type=int, default=10)
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=40)
    parser.add_argument("--trail_num_cpus", type=int, default=2)
    parser.add_argument("--trail_num_gpus", type=float, default=1)
    parser.add_argument("--trail_metric", type=str, default="bacc")
    parser.add_argument("--trail_mode", type=str, default="max")
    parser.add_argument("--exp_name", type=str, default="hyperband")
    parser.add_argument("--storage", type=str, default="~/ray_results")
    parser.add_argument("--reduction_factor", type=int, default=2)
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--swa_start_epoch", type=int, default=4)
    return parser.parse_args()

if __name__ == '__main__':
    # get_result()
    args = parse_args()
    print("=" * 40)
    print("Running with the following arguments:")
    print(json.dumps(vars(args), indent=4))
    print("=" * 40)

    init_time = time.time()
    ray.init(num_cpus=args.num_cpus, num_gpus=args.num_gpus, include_dashboard=False, configure_logging=False, logging_level=logging.ERROR)
    rs = ray.available_resources()
    print("---" * 100)
    print(f"集群可用资源：{rs}")
    print(f"初始化集群时间：{time.time() - init_time} s")
    print("---" * 100)

    start_time = time.time()
    res = hyperband_phase(args)
    total_time = time.time() - start_time
    # save_res = {"total_time": total_time, "items": res}
    print(f"总时间：{total_time} s")
    print(f"最佳配置：{res[0]}")

    res = numpy_to_python(res)
    save_result = {
        "total_time": total_time,
        "asha_num": len(res),
        "best": res[0],
        "hyperband": res
    }
    save_dict_to_file(data=save_result, base_dir=f"/data/ruipeng/workdir/autoreg/.exp_results/{args.dataset}", prefix=args.exp_name)
