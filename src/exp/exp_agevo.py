import argparse
import os
import tempfile
import time

import ray
import torch
import torch.nn as nn
from ray import tune
from ray.tune.logger import TBXLoggerCallback
from ray.tune import Tuner, TuneConfig, RunConfig

from src.data.dataloaders import get_dataloader
from src.data.meta import get_metadata
from src.data.utils import compute_class_weights
from src.exp.space import reg_space
from src.exp.util import set_seed, parse_results, save_results_json
from src.model.backbone import BackboneMLP
from src.searcher.area_searcher import AgeEvolutionSearcher
from src.trainer.trainer import Trainer

def train(config, args):
    set_seed(args.seed)
    # 数据准备
    dataset = args.dataset
    batch_size = args.batch_size
    device = args.device
    max_epochs = args.max_epochs

    meta = get_metadata(dataset=dataset)
    in_features = meta["in_features"]
    out_features = meta["out_features"]
    is_balanced = meta["is_balanced"]
    class_ratio = meta["class_ratio"]
    data_dir = meta["data_dir"]
    weights = None
    if not is_balanced:
        weights = compute_class_weights(class_ratio, method="inv")
    train_loader, valid_loader, test_loader = get_dataloader(dataset=dataset, batch_size=batch_size, data_dir=data_dir)
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
        device=device,
        reg_config=config,
    )

    checkpoint = tune.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
           trainer.load_model(os.path.join(checkpoint_dir, "checkpoint.pt"))

    for epoch in range(max_epochs):
        trainer.train(train_loader, valid_loader, epochs=1)
        loss, acc, bacc = trainer.evaluate(test_loader)
        metrics = {
            "loss": loss,
            "acc": acc,
            "bacc": bacc,
        }
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            trainer.save_model(path)
            checkpoint = tune.Checkpoint.from_directory(temp_checkpoint_dir)
            tune.report(metrics, checkpoint=checkpoint)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, default="adult")
    parser.add_argument("--batch_size", type=int, required=True, default=64)
    parser.add_argument("--seed", type=int, required=True, default=42)
    parser.add_argument("--device", type=str, required=True, default="cpu")
    parser.add_argument("--num_cpus", type=int, required=True, default=4)
    parser.add_argument("--num_gpus", type=int, required=True, default=1)
    parser.add_argument("--max_concurrent_trials", type=int, required=True, default=8)
    parser.add_argument("--lr", type=float, required=True, default=1e-3)
    parser.add_argument("--momentum", type=float, required=True, default=0.9)
    parser.add_argument("--max_epochs", type=int, required=True, default=10)
    parser.add_argument("--num_samples", type=int, required=True, default=500)
    parser.add_argument("--trail_num_cpus", type=int, required=True, default=1)
    parser.add_argument("--trail_num_gpus", type=float, required=True, default=0.1)
    parser.add_argument("--trail_metric", type=str, required=True, default="bacc")
    parser.add_argument("--trail_mode", type=str, required=True, default="max")
    parser.add_argument("--exp_name", type=str, required=True, default="default")
    parser.add_argument("--storage", type=str, default="~/ray_results")
    parser.add_argument("--reduction_factor", type=int, required=True, default=3)
    parser.add_argument("--population_size", type=int, required=True, default=10)
    parser.add_argument("--sample_size", type=int, required=True, default=3)
    return parser.parse_args()

if __name__ == '__main__':
    start_time = time.time()
    args = parse_args()
    set_seed(args.seed)
    ray.init(num_cpus=args.num_cpus, num_gpus=args.num_gpus)

    searcher = AgeEvolutionSearcher(
        search_space=reg_space,
        population_size=args.population_size,
        sample_size=args.sample_size,
        metric=args.trail_metric,
        mode=args.trail_mode,
    )

    tuner = Tuner(
        trainable=tune.with_resources(
            tune.with_parameters(train, args=args),
            resources={"cpu": args.trail_num_cpus, "gpu": args.trail_num_gpus}

        ),
        tune_config=TuneConfig(
            search_alg=searcher,
            num_samples=args.num_samples,

        ),
        run_config=RunConfig(
            # callbacks=[TBXLoggerCallback()],
            name=args.exp_name,
            storage_path=args.storage,
        )
    )
    results = tuner.fit()
    end_time = time.time()
    df = results.get_dataframe()
    print(df)
    res = parse_results(df)
    print(f"总时间：{end_time - start_time}")
    save_results_json(res, args.exp_name, "/home/zrp/pycharmProjects/autoreg/.exp_results/")

# python exp_agevo.py --dataset adult --batch_size 64 --seed 42 --device cuda --num_cpus 4 --num_gpus 1 --max_concurrent_trials 4 --lr 1e-3 --momentum 0.9 --max_epochs 4 --num_samples 10 --trail_num_cpus 1 --trail_num_gpus 0.1 --trail_metric bacc --trail_mode max --exp_name agevo --storage ~/ray_results --reduction_factor 2 --population_size 10 --sample_size 3
