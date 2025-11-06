import random
import time

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, Callable
from timm.optim import Lookahead

from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from src.data.dataset.adult import get_adult_dataloader_sampled
from src.data.meta import get_metadata
from src.data.utils import compute_class_weights
from src.model.backbone import BackboneMLP
from src.regular.data_augment import mixup, cutout, fgsm, cutmix
from src.regular.weight_decay import weight_decay_regular


class StepTrainer(object):
    def __init__(self,
        model: nn.Module,
        criterion: Optional[nn.Module] = None,
        optimizer_name: str = "AdamW", # Literal["SGD", "AdamW", "rmsprop"]
        lr:float = 1e-3,
        momentum=0.9,
        device: str = "cpu",
        reg_config: Optional[dict] = None,
    ):
        self.reg_config = reg_config
        self.device = torch.device("cuda" if torch.cuda.is_available() and device=="cuda" else "cpu")
        self.model = model.to(self.device)
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = self._init_optimizer(optimizer_name, lr, momentum)
        self.lr = lr
        self.momentum = momentum
        self.scheduler = None
        self.augment_fn = self._init_data_augment()
        self.swa_model = None
        self.swa_scheduler = None
        self.swa_is_active = False


    def _init_optimizer(self, name: str, lr: float, momentum: float = 0.9):
        if name == "SGD":
            optimizer =  torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        elif name == "AdamW":
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        elif name == "rmsprop":
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr, momentum=momentum)
        else:
            raise ValueError(f"Unsupported optimizer: {name}")
        # 包装 lookahead
        if self.reg_config and self.reg_config.get("use_lookahead", False):
            optimizer = Lookahead(optimizer, alpha=self.reg_config.get("lookahead_alpha", 0.5), k=self.reg_config.get("lookahead_step", 5))
        return optimizer

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
        # if not self.swa_is_active:
            # print(f"[SWA] Initialized at epoch {self.start_epoch + 1}")
        self.swa_is_active = True

    def train(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            max_steps: int = 300,
            val_interval: int = 300,
            verbose: bool = True,
    ):
        if self.scheduler is None:
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=301,
                T_mult=2,
                eta_min=1e-5,
                last_epoch=-1
            )
        self.model.train()
        results = {"step": [], "train_loss": [], "train_bacc": [], "val_loss": [], "val_bacc": []}
        train_loss = 0
        train_correct = 0
        train_total = 0
        num_classes, conf = None, None  # 矩阵 行=真实标签，列=预测
        # 取 batch；到头就循环 dataloader
        train_iter = iter(train_loader)
        for step in range(1, 1+max_steps):
            # 激活 SWA
            if self.reg_config.get("use_swa", False) and not self.swa_is_active and step >= self.reg_config.get("swa_start_epoch", 200):
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
            train_loss += ce_loss.item()
            _, pred = logits.max(1)
            train_correct += pred.eq(y).sum().item()
            train_total += y.size(0)
            # 统计混淆矩阵，用于 Balanced Accuracy
            if num_classes is None:
                num_classes = logits.size(1)
                conf = torch.zeros((num_classes, num_classes), dtype=torch.long)
            y_cpu, p_cpu = y.detach().cpu(), pred.detach().cpu()
            idx = y_cpu * num_classes + p_cpu
            conf.view(-1).index_add_(0, idx, torch.ones_like(idx, dtype=torch.long))

            if self.swa_is_active:
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()

            if step % val_interval == 0:
                train_loss = train_loss / val_interval
                # train_acc = train_correct / train_total
                row_sum = conf.sum(dim=1).clamp_min(1)
                train_bal_acc = (conf.diag().float() / row_sum.float()).mean().item()
                val_loss, val_acc, val_bacc = self.evaluate(val_loader)
                if verbose:
                    cur_lr = self.optimizer.param_groups[0]['lr']
                    print(f"Step {step} | Train Loss={train_loss:.6f} | Train BAcc={train_bal_acc:.6f} | Val Loss={val_loss:.6f} | Val BAcc={val_bacc:.6f} | LR: {cur_lr:.6f}")
                results["step"].append(step)
                results["train_loss"].append(train_loss)
                results["train_bacc"].append(train_bal_acc)
                results["val_loss"].append(val_loss)
                results["val_bacc"].append(val_bacc)

                train_loss = 0
                train_correct = 0
                train_total = 0
                num_classes, conf = None, None  # 矩阵 行=真实标签，列=预测
            # 验证集测试
            # —— 推进学习率调度（与 SWA 互斥）——
            if not self.swa_is_active and self.scheduler is not None:
                self.scheduler.step()

        # 更新 BN stats
        if self.swa_is_active:
            if any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) for m in self.swa_model.modules()):
                update_bn(train_loader, self.swa_model, device=self.device)

        return results


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
        "use_data_augment": True,
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

    train_loader, valid_loader, test_loader = get_adult_dataloader_sampled(data_dir=data_dir, batch_size=batch_size, sample_ratio=0.2)

    # 初始化模型
    hidden_features = [512, 512, 512, 512, 512, 512]
    config = {}
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
        optimizer_name="AdamW",
        lr=1e-3,
        momentum=0.9,
        device=device,
        reg_config=config,
    )
    start_time = time.time()
    result = trainer.train(train_loader, valid_loader, 300, 10, True)
    end_time = time.time()
    print(f"spend_time: {end_time - start_time}")
    plot_results(result)









