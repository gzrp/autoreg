import random
import torch
import numpy as np
from typing import Tuple
from torch import nn


def cutout(x: torch.Tensor, y: torch.Tensor,
           cutout_ratio: float = 0.1, prob: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """ cut-out 数据增强: 随机遮挡部分特征，但不混合标签，因此混合系数 为 1.0
    :param x:               特征张量， shape (B, F)
    :param y:               标签张量， shape (B,)
    :param cutout_ratio:    cutout 率  0.0~0.5  默认 0.1
    :param prob:            触发概率    0.0~1.0  默认 0.5
    :return:                增强张量, 原标签, 混合标签, 混合系数
    """
    p = random.random()
    if p > prob:
        return x, y, y, 1.0
    x = x.clone()
    batch_size, num_features = x.size()
    cut_len = int(num_features * cutout_ratio)
    for i in range(batch_size):
        idx = torch.randperm(num_features)[:cut_len]
        x[i, idx] = 0.0
    return x, y, y, 1.0

def mixup(x: torch.Tensor, y: torch.Tensor,
          alpha: float = 0.2, prob: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """mix-up 数据增强：按照比例混合两个样本
    :param x:                   特征张量， shape (B, F)
    :param y:                   标签张量， shape (B,)
    :param alpha:               范围在 0.0~1.0 之间，值越大，混合的越充分
    :param prob:                概率   0.0~1.0
    :return:                    增强张量, 原标签, 混合标签, 混合系数
    """
    p = random.random()
    if p > prob:
        return x, y, y, 1.0
    x = x.clone()
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y.clone(), y[index].clone()
    return mixed_x, y_a, y_b, lam

def cutmix(x: torch.Tensor, y: torch.Tensor,
        alpha: float = 0.2, prob: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    cutmix 数据增强：将一个样本的部分特征区域替换为另一个样本对应区域的特征，同时对标签按相应比例进行混合。
    :param x:                   特征张量， shape (B, F)
    :param y:                   标签张量， shape (B,)
    :param alpha:               范围在 0~1.0 之间，值越大，混合的越充分
    :param prob:                概率  0.0~1.0
    :return:                    增强张量, 原标签, 混合标签, 混合系数
    """
    p = random.random()
    if p > prob:
        return x, y, y, 1.0
    x = x.clone()
    lam = np.random.beta(alpha, alpha)
    batch_size, num_features = x.size()
    index = torch.randperm(batch_size).to(x.device)
    num_cut = int(num_features * (1 - lam))
    selected_cols = np.random.choice(num_features, num_cut, replace=False)
    # 替换特征列
    x[:, selected_cols] = x[index][:, selected_cols]
    # 标签混合比例（按被替换列数量估算）
    lam_adjusted = 1.0 - num_cut / num_features
    y_a, y_b = y.clone(), y[index].clone()
    return x, y_a, y_b, lam_adjusted

def fgsm(model: nn.Module, x: torch.Tensor, y: torch.Tensor, loss_fn: nn.Module,
         epsilon: float = 0.1, prob: float = 0.5, col: list = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """对输入特征进行 FGSM 增强（不会污染模型参数的梯度），适用于表格数据。
    :param model:      模型（不影响其权重梯度）
    :param x:          输入特征，shape (B, F)
    :param y:          标签，shape (B,)
    :param loss_fn:    损失函数
    :param epsilon:    扰动强度 0.0~0.3     超参数
    :param prob        触发概率 0.0~1.0
    :param col         应用列
    :return:           对抗样本 x_adv，原始标签 y
    """
    p = random.random()
    if p > prob or not col:
        return x, y, y, 1.0
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    x_adv = x.clone().detach().requires_grad_(True).to(device)
    y = y.clone().to(device)
    with torch.enable_grad():
        outputs = model(x_adv)
        loss = loss_fn(outputs, y)
        # 只对输入 x 求导，不影响 model 参数
        grad = torch.autograd.grad(
            outputs=loss,
            inputs=x_adv,
            retain_graph=False,
            create_graph=False
        )[0]
        # 仅对连续列添加扰动
        if col is not None:
            mask = torch.zeros_like(grad)
            mask[:, col] = 1
            grad = grad * mask
        x_adv = x_adv + epsilon * grad.sign()
        x_adv = x_adv.detach()
    # 恢复原始模式
    model.train(was_training)
    return x_adv, y, y, 1.0
