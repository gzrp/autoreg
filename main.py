import random
import torch
import numpy as np
from torch import nn

from src.data.dataloaders import get_dataloader
from src.data.meta import get_metadata
from src.data.utils import compute_class_weights
from src.model.backbone import BackboneMLP
from src.trainer.trainer import Trainer


# 设置随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed = 42
set_seed(seed)

# 加载数据集
meta = get_metadata(dataset="adult")
in_features = meta["in_features"]
out_features = meta["out_features"]
is_balanced = meta["is_balanced"]
class_ratio = meta["class_ratio"]
data_dir = meta["data_dir"]
batch_size = meta["batch_size"]
weights = None
if not is_balanced:
    weights = compute_class_weights(class_ratio, method="inv")

hidden_features = [512, 512, 512, 512, 512, 512]
train_loader, valid_loader, test_loader = get_dataloader(dataset="adult", batch_size=batch_size, data_dir=data_dir)
# 正则化配置

regular = {
    "use_l1": False,  # 权重衰减配置
    "l1_lambda": 0.00,
    "use_l2": False,
    "l2_lambda": 0.00,
    "use_dropout": False,  # dropout配置
    "drop_rate": 0.00,
    "use_bn": False,  # bn 配置
    "use_ln": False,
    "use_skip": False,  # skip 配置
    "skip_type": "None",
    "skip_step": 1,
    "skip_drop_prob": 0.0,
    "use_data_augment": False,  # 数据增强配置
    "da_type": "None",
    "cutout_ratio": 0.0,
    "cutout_prob": 0.0,
    "mixup_alpha": 0.0,
    "mixup_prob": 0.0,
    "cutmix_alpha": 0.0,
    "cutmix_prob": 0.0,
    "fgsm_epsilon": 0.0,
    "fgsm_prob": 0.0,
    "use_swa": False,  # swa 配置
    "use_lookahead": False,  # lookahead 配置
}

# 初始化模型
model = BackboneMLP(
    input_dim=in_features,
    hidden_dims=hidden_features,
    output_dim=out_features,
    reg_config=regular,
)

# 初始化训练器
device = "cuda" if torch.cuda.is_available() else "cpu"
if weights is not None:
    ce_weight = torch.tensor(weights, dtype=torch.float32).to(torch.device(device))
else:
    ce_weight = None

criterion = nn.CrossEntropyLoss(weight=ce_weight)
trainer = Trainer(
    model=model,
    criterion=criterion,
    optimizer_name="AdamW",
    lr= 1e-3,
    momentum=0.9,
    device=device,
    reg_config=regular,
)

# 执行训练

if __name__ == '__main__':
    trainer.train(
        train_loader=train_loader,
        val_loader=valid_loader,
        epochs=40,
        verbose=True,
        save_path="a.pth"
    )
    # trainer.print()
    trainer.load_model("a.pth")
    # trainer.print()

    trainer.train(
        train_loader=train_loader,
        val_loader=valid_loader,
        epochs=80,
        verbose=True,
        save_path="b.pth"
    )
    trainer.print()


    # loss, acc, bacc = trainer.evaluate(test_loader)
    # print(loss)
    # print(acc)
    # print(bacc)
