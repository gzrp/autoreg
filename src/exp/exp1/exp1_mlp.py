import time
import torch
from torch import nn

from src.data.dataloaders import get_dataloader
from src.data.meta import get_metadata
from src.data.utils import compute_class_weights
from src.exp.exp1.util import set_seed
from src.model.backbone import BackboneMLP
from src.trainer.trainer import Trainer

if __name__ == '__main__':
    start_time = time.time()
    config = {
        "use_l1": False,
        "l1_lambda": 0.00,
        "use_l2": False,
        "l2_lambda": 0.00,
        "use_dropout": False,
        "drop_rate": 0.0,
        "use_bn": False,
        "use_ln": False,
        "use_skip": False,
        "skip_type": "normal",
        "skip_step": 1,
        "skip_drop_prob": 0.0,
        "use_data_augment": False,
        "da_type": "mixup",
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

    set_seed(42)
    # 数据准备
    dataset="frappe"
    meta = get_metadata(dataset=dataset)
    in_features = meta["in_features"]
    out_features = meta["out_features"]
    is_balanced = meta["is_balanced"]
    class_ratio = meta["class_ratio"]
    data_dir = meta["data_dir"]
    batch_size = meta["batch_size"]
    weights = None
    if not is_balanced:
        weights = compute_class_weights(class_ratio, method="inv")

    train_loader, valid_loader, test_loader = get_dataloader(dataset=dataset, data_dir=data_dir, batch_size=batch_size)

    # 初始化模型
    hidden_features = [512, 512, 512, 512, 512, 512]

    model = BackboneMLP(
        input_dim=in_features,
        hidden_dims=hidden_features,
        output_dim=out_features,
        reg_config=config,
    )
    # 初始化训练器
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    if weights is not None:
        ce_weight = torch.tensor(weights, dtype=torch.float32).to(torch.device(device))
    else:
        ce_weight = None

    criterion = nn.CrossEntropyLoss(weight=ce_weight)
    trainer = Trainer(
        model=model,
        criterion=criterion,
        lr=1e-3,
        device=device,
        reg_config=config,
    )

    result = []
    for epoch in range(16):
        trainer.train(train_loader, valid_loader, epochs=1, verbose=True)
        loss, acc, bacc = trainer.evaluate(test_loader)
    loss, acc, bacc = trainer.evaluate(test_loader)
    print(bacc)
    print(time.time() - start_time)

