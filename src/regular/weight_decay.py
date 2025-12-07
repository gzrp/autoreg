import torch
from torch import nn

def weight_decay_regular_v0(
        model: nn.Module,
        l1_lambda: float = 0.0,
        l2_lambda: float = 0.0,
) -> torch.Tensor:
    """
    计算模型的 weight_decay 正则项
    :param model: 模型
    :param l1_lambda: L1 正则化系数，默认 0.0 表示不启用
    :param l2_lambda: L2 正则化系数，默认 0.0 表示不启用
    :return: 正则化 lose
    """
    l1_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    l2_loss = torch.tensor(0.0, device=next(model.parameters()).device)

    for param in model.parameters():
        if l1_lambda > 0:
            l1_loss += torch.sum(torch.abs(param))
        if l2_lambda > 0:
            l2_loss += torch.sum(param ** 2)
    return l1_lambda * l1_loss + l2_lambda * l2_loss

def weight_decay_regular(
        model: nn.Module,
        l1_lambda: float = 0.0,
        l2_lambda: float = 0.0,
) -> torch.Tensor:
    """
    计算模型的 weight_decay 正则项
    :param model: 模型
    :param l1_lambda: L1 正则化系数，默认 0.0 表示不启用
    :param l2_lambda: L2 正则化系数，默认 0.0 表示不启用
    :return: 正则化 lose
    """
    params = [p.view(-1) for p in model.parameters() if p.requires_grad]
    if not params:
        return torch.tensor(0.0, device=next(model.parameters()).device)

    flat = torch.cat(params)

    loss = 0.0
    if l1_lambda > 0:
        loss += l1_lambda * flat.abs().sum()
    if l2_lambda > 0:
        loss += l2_lambda * (flat ** 2).sum()
    return loss

