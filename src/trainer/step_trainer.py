import random
import time

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, Callable
from timm.optim import Lookahead

from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

from src.data.dataset.adult import get_adult_dataloader
from src.data.meta import get_metadata
from src.data.utils import compute_class_weights
from src.model.backbone import BackboneMLP
from src.regular.data_augment import mixup, cutout, fgsm, cutmix
from src.regular.weight_decay import weight_decay_regular


class StepTrainer(object):
    def __init__(self,
        model: nn.Module,
        criterion: Optional[nn.Module] = None,
        lr:float = 1e-3,
        device: str = "cpu",
        swa_start_epoch: int = 2,
        reg_config: Optional[dict] = None,
    ):
        self.reg_config = reg_config
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.criterion = criterion
        self.lr = lr
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        if self.reg_config.get("use_lookahead", False):
            self.optimizer = Lookahead(optimizer)
        else:
            self.optimizer = optimizer

        self.augment_fn = self._init_data_augment()
        self.swa_model = None
        self.swa_scheduler = None
        self.swa_is_active = False
        self.swa_start_epoch = swa_start_epoch

    def _init_data_augment(self) -> Optional[Callable]:
        """根据正则化配置初始化数据增强函数"""
        if not self.reg_config or not self.reg_config.get("use_data_augment", False):
            return None
        if self.reg_config.get("da_type") == 'mixup':
            def augment_fn(model, x, y, col):
                return mixup(x, y, alpha=self.reg_config.get("mixup_alpha"), prob=self.reg_config.get("mixup_prob"))
        elif self.reg_config.get("da_type") == 'cutmix':
            def augment_fn(model, x, y, col):
                return cutmix(x, y, alpha=self.reg_config.get("cutmix_alpha"), prob=self.reg_config.get("cutmix_prob"))
        elif self.reg_config.get("da_type") == 'cutout':
            def augment_fn(model, x, y, col):
                return cutout(x, y, cutout_ratio=self.reg_config.get("cutout_ratio"), prob=self.reg_config.get("cutout_prob"))
        elif self.reg_config.get("da_type") == 'fgsm':
            def augment_fn(model, x, y, col):
                return fgsm(model, x, y, loss_fn=self.criterion, epsilon=self.reg_config.get("fgsm_epsilon"), prob=self.reg_config.get("fgsm_prob"), col=col)
        else:
            return None
        return augment_fn

    def _apply_weight_decay(self, loss: torch.Tensor) -> torch.Tensor:
        use_l1 = self.reg_config.get("use_l1", False)
        use_l2 = self.reg_config.get("use_l2", False)
        if use_l1 or use_l2:
            l1_lambda = self.reg_config.get("l1_lambda", 0.0) if use_l1 else 0.0
            l2_lambda = self.reg_config.get("l2_lambda", 0.0) if use_l2 else 0.0
            wd_loss = weight_decay_regular(self.model, l1_lambda=l1_lambda, l2_lambda=l2_lambda)
            return loss + wd_loss
        return loss

    def _init_swa(self):
        self.swa_model = AveragedModel(self.model)
        self.swa_scheduler = SWALR(optimizer=self.optimizer, swa_lr=self.reg_config.get("swa_lr", 0.001))
        self.swa_is_active = True

    def train(
            self,
            train_loader: DataLoader,
            max_steps: int = 300,
    ):
        self.model.train()
        # 取 batch；到头就循环 dataloader
        train_iter = iter(train_loader)
        for step in range(1, 1+max_steps):
            # 激活 SWA
            if self.reg_config.get("use_swa", False) and not self.swa_is_active and step >= self.swa_start_epoch:
                self._init_swa()
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            # 应用数据增强
            if self.augment_fn is not None:
                col = train_loader.dataset.feature_indices.get("numerical")
                x, y_a, y_b, lam = self.augment_fn(self.model, x, y, col)
            else:
                y_a = y_b = y
                lam = 1.0

            # 增强之后再统一移到 device
            x = x.to(self.device)
            y = y.to(self.device)
            y_a = y_a.to(self.device)
            y_b = y_b.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(x)
            ce_loss = lam * self.criterion(logits, y_a) + (1 - lam) * self.criterion(logits, y_b)
            loss = self._apply_weight_decay(ce_loss)

            loss.backward()
            self.optimizer.step()
            # 统计混淆矩阵，用于 Balanced Accuracy
            if self.swa_is_active:
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()

        # 更新 BN stats
        if self.swa_is_active:
            if any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) for m in self.swa_model.modules()):
                update_bn(train_loader, self.swa_model, device=self.device)


    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float, float]:
        if self.reg_config.get("use_swa") and self.swa_is_active:
            self.swa_model.eval()
            model = self.swa_model
        else:
            self.model.eval()
            model = self.model
        total_loss, correct, total = 0.0, 0, 0
        num_classes, conf = None, None  # 行=真实标签，列=预测
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                logits = model(x)
                loss = self.criterion(logits, y)
                total_loss += loss.item()
                _, pred = logits.max(1)
                correct += pred.eq(y).sum().item()
                total += y.size(0)
                # 初始化混淆矩阵
                if num_classes is None:
                    num_classes = logits.size(1)
                    conf = torch.zeros((num_classes, num_classes), dtype=torch.long)
                y_cpu, p_cpu = y.detach().cpu(), pred.detach().cpu()
                idx = y_cpu * num_classes + p_cpu
                conf.view(-1).index_add_(0, idx, torch.ones_like(idx, dtype=torch.long))
        # Balanced Accuracy = 平均每类召回率 = mean(diag(conf) / row_sum)
        row_sum = conf.sum(dim=1).clamp_min(1)
        bal_acc = (conf.diag().float() / row_sum.float()).mean().item()
        return total_loss / len(dataloader), correct / total, bal_acc

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

import matplotlib.pyplot as plt

def plot_results(results):
    steps = results["step"]
    train_loss = results["train_loss"]
    train_bacc = results["train_bacc"]
    val_loss = results["val_loss"]
    val_bacc = results["val_bacc"]

    # 创建画布
    plt.figure(figsize=(12, 5))

    # -------- LOSS 曲线 --------
    plt.subplot(1, 2, 1)
    plt.plot(steps, train_loss, marker='o', label="Train Loss")
    plt.plot(steps, val_loss, marker='s', label="Val Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.grid(True)
    plt.legend()

    # -------- Balanced Accuracy 曲线 --------
    plt.subplot(1, 2, 2)
    plt.plot(steps, train_bacc, marker='o', label="Train BAcc")
    plt.plot(steps, val_bacc, marker='s', label="Val BAcc")
    plt.xlabel("Steps")
    plt.ylabel("Balanced Accuracy")
    plt.title("Training vs Validation Balanced Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    config = {
        "use_l1": False,
        "l1_lambda": 0.0,
        "use_l2": False,
        "l2_lambda": 0.0,
        "use_dropout": False,
        "drop_rate": 0.0,
        "use_bn": False,
        "use_ln": False,
        "use_skip": False,
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

    set_seed(42)
    # 数据准备
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

    train_loader, valid_loader, test_loader = get_adult_dataloader(data_dir=data_dir, batch_size=batch_size)

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
    if weights is not None:
        ce_weight = torch.tensor(weights, dtype=torch.float32).to(torch.device(device))
    else:
        ce_weight = None

    criterion = nn.CrossEntropyLoss(weight=ce_weight)
    trainer = StepTrainer(
        model=model,
        criterion=criterion,
        lr=1e-3,
        device=device,
        reg_config=config,
    )
    start_time = time.time()

    trainer.train(train_loader, max_steps=300)
    loss, acc, bacc = trainer.evaluate(test_loader)
    print(loss, acc, bacc , time.time() - start_time)
    # plot_results(result)

