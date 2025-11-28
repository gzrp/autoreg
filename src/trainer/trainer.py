import time
import torch
import torch.nn as nn
from typing import Optional, Tuple, Callable
from timm.optim import Lookahead
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

from src.data.dataset.frappe import get_frappe_dataloader
from src.data.meta import get_metadata
from src.data.utils import compute_class_weights
from src.exp.exp1.util import set_seed
from src.model.backbone import BackboneMLP
from src.regular.data_augment import mixup, cutout, fgsm, cutmix
from src.regular.weight_decay import weight_decay_regular


class Trainer(object):
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

        # 记录日志
        self.start_epoch = 0
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.train_bacc_history = []
        self.val_bacc_history = []

        self.scheduler = None

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

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float, float]:
        """
        训练一轮
        :param dataloader: 训练集
        :return: 损失，准去率，平衡准确率
        """
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        num_classes, conf = None, None  # 矩阵 行=真实标签，列=预测
        for x, y in dataloader:
            # start_time = time.time()
            # 应用数据增强
            if self.augment_fn is not None:
                col = dataloader.dataset.feature_indices.get("numerical")
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
            loss = lam * self.criterion(logits, y_a) + (1 - lam) * self.criterion(logits, y_b)
            loss = self._apply_weight_decay(loss)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, pred = logits.max(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
            # 统计混淆矩阵，用于 Balanced Accuracy
            if num_classes is None:
                num_classes = logits.size(1)
                conf = torch.zeros((num_classes, num_classes), dtype=torch.long)
            y_cpu, p_cpu = y.detach().cpu(), pred.detach().cpu()
            idx = y_cpu * num_classes + p_cpu
            conf.view(-1).index_add_(0, idx, torch.ones_like(idx, dtype=torch.long))

        # Balanced Accuracy = 平均每类召回率 = mean(diag(conf) / row_sum)
        row_sum = conf.sum(dim=1).clamp_min(1)
        bal_acc = (conf.diag().float() / row_sum.float()).mean().item()
        self.start_epoch += 1
        return total_loss / len(dataloader), correct / total, bal_acc

    def _init_swa(self):
        self.swa_model = AveragedModel(self.model)
        self.swa_scheduler = SWALR(optimizer=self.optimizer, swa_lr=self.reg_config.get("swa_lr", 0.001))
        self.swa_is_active = True

    def train(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int = 10,
            verbose: bool = True,
            save_path: Optional[str] = None,
    ):
        if self.scheduler is None:
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=101,
                T_mult=2,
                eta_min=1e-6,
                last_epoch=self.start_epoch-1
            )
        for epoch in range(self.start_epoch+1, self.start_epoch+epochs+1):
            start_time = time.time()
            # 激活 SWA
            if self.reg_config.get("use_swa", False) and not self.swa_is_active and epoch >= self.swa_start_epoch:
                self._init_swa()

            train_loss, train_acc, train_bacc = self.train_epoch(train_loader)
            self.train_loss_history.append(train_loss)
            self.train_acc_history.append(train_acc)
            self.train_bacc_history.append(train_bacc)

            if self.swa_is_active:
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()

            # 验证集测试
            val_loss, val_acc, val_bacc = self.evaluate(val_loader)
            self.val_loss_history.append(val_loss)
            self.val_acc_history.append(val_acc)
            self.val_bacc_history.append(val_bacc)

            # —— 推进学习率调度（与 SWA 互斥）——
            if not self.swa_is_active and self.scheduler is not None:
                self.scheduler.step()

            # 打印当前学习率
            cur_lr = self.optimizer.param_groups[0]['lr']
            if verbose:
                print(f"[Epoch {epoch}] Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.6f}, Train BAcc: {train_bacc} | "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.6f}, Val BAcc: {val_bacc:.6f} | LR: {cur_lr:.6f} | Time: {time.time() - start_time:.2f}")

        # 更新 BN stats
        if self.swa_is_active:
            if any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) for m in self.swa_model.modules()):
                update_bn(train_loader, self.swa_model, device=self.device)

        if save_path is not None:
            self.save_model(save_path)

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float, float]:
        if self.reg_config.get("use_swa") and self.swa_is_active:
            self.swa_model.eval()
            model = self.swa_model
        else:
            self.model.eval()
            model = self.model
        model.eval()
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

    def save_model(self, path: str):
        state = {
            "epoch": len(self.train_loss_history),
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "train_loss_history": self.train_loss_history,
            "train_acc_history": self.train_acc_history,
            "train_bacc_history": self.train_bacc_history,
            "val_loss_history": self.val_loss_history,
            "val_acc_history": self.val_acc_history,
            "val_bacc_history": self.val_bacc_history,
        }
        if self.swa_is_active:
            state["swa_model_state"] = self.swa_model.state_dict()
        torch.save(state, path)
        # self._logger(f"Model saved to {path}")

    def load_model(self, path: str):
        cpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(cpt["model_state"])
        self.model.to(self.device)
        self.optimizer.load_state_dict(cpt["optimizer_state"])
        self.start_epoch = cpt.get("epoch", 0)
        self.train_loss_history = cpt.get("train_loss_history", [])
        self.train_acc_history = cpt.get("train_acc_history", [])
        self.train_bacc_history = cpt.get("train_bacc_history", [])
        self.val_loss_history = cpt.get("val_loss_history", [])
        self.val_acc_history = cpt.get("val_acc_history", [])
        self.val_bacc_history = cpt.get("val_bacc_history", [])
        if self.reg_config.get("use_swa", False) and "swa_model_state" in cpt:
            self._init_swa()
            self.swa_model.load_state_dict(cpt["swa_model_state"])



if __name__ == '__main__':
    # {"best_config": {"use_l1": true, "l1_lambda": 0.001, "use_l2": false, "l2_lambda": 0.0, "use_dropout": false, "drop_rate": 0.0, "use_bn": false, "use_ln": true, "use_skip": true, "skip_type": "normal", "skip_step": 1, "skip_drop_prob": 0.0, "use_data_augment": false, "da_type": "None", "cutout_ratio": 0.0, "cutout_prob": 0.0, "mixup_alpha": 0.0, "mixup_prob": 0.0, "cutmix_alpha": 0.0, "cutmix_prob": 0.0, "fgsm_epsilon": 0.0, "fgsm_prob": 0.0, "use_swa": true, "use_lookahead": true}}
    config = {
        "use_l1": False,
        "l1_lambda": 0.00,
        "use_l2": False,
        "l2_lambda": 0.00,
        "use_dropout": True,
        "drop_rate": 0.2,
        "use_bn": True,
        "use_ln": False,
        "use_skip": True,
        "skip_type": "normal",
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
    meta = get_metadata(dataset="frappe")
    in_features = meta["in_features"]
    out_features = meta["out_features"]
    is_balanced = meta["is_balanced"]
    class_ratio = meta["class_ratio"]
    data_dir = meta["data_dir"]
    batch_size = meta["batch_size"]
    weights = None
    if not is_balanced:
        weights = compute_class_weights(class_ratio, method="inv")

    train_loader, valid_loader, test_loader = get_frappe_dataloader(data_dir=data_dir, batch_size=batch_size)
    start_time = time.time()
    # 初始化模型
    hidden_features = [512, 512, 512, 512, 512, 512]
    # config = {}
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
    trainer = Trainer(
        model=model,
        criterion=criterion,
        lr=1e-3,
        device=device,
        reg_config=config,
    )

    for epoch in range(20):
        trainer.train(train_loader, valid_loader, epochs=1, verbose=True)

    loss, acc, bacc = trainer.evaluate(test_loader)
    print("loss:", loss)
    print("acc:", acc)
    print("bacc:", bacc)
    print("time: ", time.time() - start_time)






