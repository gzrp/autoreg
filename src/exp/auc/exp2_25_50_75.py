import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.datasets import get_dataset
from src.data.meta import get_metadata
from src.data.utils import compute_class_weights
from src.model.backbone import BackboneMLP
from src.trainer.trainer_new import Trainer


def phase2_one(config, dataset, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    # 数据准备
    meta = get_metadata(dataset=dataset)
    in_features = meta["in_features"]
    out_features = meta["out_features"]
    is_balanced = meta["is_balanced"]
    class_ratio = meta["class_ratio"]
    data_dir = meta["data_dir"]
    batch_size = meta["batch_size"]

    train_set, val_set, test_set = get_dataset(dataset, data_dir)

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
    device = "cuda:3" if torch.cuda.is_available() else "cpu"
    ce_weight = None
    if not is_balanced:
        weights = compute_class_weights(class_ratio, method="inv")
        ce_weight = torch.tensor(weights, dtype=torch.float32).to(torch.device(device))

    criterion = nn.CrossEntropyLoss(weight=ce_weight)
    trainer = Trainer(
        model=model,
        criterion=criterion,
        lr=1e-3,
        swa_start_epoch=2,
        device=device,
        reg_config=config,
        metric_type="AUC"
    )
    for epoch in range(8):
        trainer.train(train_loader, valid_loader, epochs=1, verbose=False)
        loss, acc, auc = trainer.evaluate(test_loader)
    auc_history = trainer.test_auc_history,
    return auc_history


if __name__ == '__main__':
    start_time = time.time()
    config = {
        "use_l1": False,
        "l1_lambda":0.0,
        "use_l2": False,
        "l2_lambda": 0.00,
        "use_dropout": False,
        "drop_rate": 0.0,
        "use_bn": False,
        "use_ln": False,
        "use_skip": False ,
        "skip_type": "None",
        "skip_step": 1,
        "skip_drop_prob": 0.0,
        "use_data_augment": False,
        "da_type": "None",
        "cutout_ratio": 0.0,
        "cutout_prob": 0.0,
        "mixup_alpha": 0.0,
        "mixup_prob": 0.0,
        "cutmix_alpha": 0.0,
        "cutmix_prob": 0.0,
        "fgsm_epsilon": 0.0,
        "fgsm_prob": 0.0,
        "use_swa": False,
        "use_lookahead": False,
    }

    seeds = [42, 2023, 7, 1234, 11]
    dataset = "adult"

    all_results = []
    for seed in seeds:
        metrics = phase2_one(config=config, dataset=dataset, seed=seed)
        all_results.append(metrics[0])

    al_results = np.array(all_results)
    q25 = np.percentile(all_results, 25, axis=0)
    q50 = np.percentile(all_results, 50, axis=0)
    q75 = np.percentile(all_results, 75, axis=0)
    print("q25:", q25)
    print("q50:", q50)
    print("q75:", q75)