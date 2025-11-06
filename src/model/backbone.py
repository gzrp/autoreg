import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union

class BackboneMLP(nn.Module):
    def __init__(self,
                 input_dim: int = 10,
                 hidden_dims: List[int] = (512, 512, 512, 512, 512),
                 output_dim: int = 10,
                 reg_config: dict = None):
        super(BackboneMLP, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.use_dropout = reg_config['use_dropout'] if reg_config else False
        self.drop_rate = reg_config['drop_rate'] if self.use_dropout else 0.0

        self.use_bn = reg_config['use_bn'] if reg_config else False
        self.use_ln = reg_config['use_ln'] if reg_config else False

        self.use_skip = reg_config['use_skip'] if reg_config else False
        self.skip_type = reg_config['skip_type'] if self.use_skip else "None"
        self.skip_step = reg_config['skip_step'] if self.use_skip else 1
        self.skip_drop_prob = reg_config['skip_drop_prob'] if self.use_skip and self.skip_type == "random" else 0.0

        dims : List[int] = [input_dim] + hidden_dims

        self.layers: nn.ModuleList = nn.ModuleList()
        self.bns: nn.ModuleList = nn.ModuleList()
        self.lns: nn.ModuleList = nn.ModuleList()
        self.dropouts: nn.ModuleList = nn.ModuleList()

        for i in range(len(hidden_dims)):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            if self.use_bn:
                self.bns.append(nn.BatchNorm1d(dims[i+1]))

            if self.use_ln:
                self.lns.append(nn.LayerNorm(dims[i+1]))

            if self.use_dropout:
                if isinstance(self.drop_rate, float):
                    p = self.drop_rate
                else:
                    assert len(self.drop_rate) == len(hidden_dims), \
                        "drop rate list 长度应与 hidden_dims 匹配"
                    p = self.drop_rate[i]
                self.dropouts.append(nn.Dropout(p))

        self.output_layer: nn.Linear = nn.Linear(dims[-1], output_dim)

    def forward(self, x) -> torch.Tensor:
        residual = None
        for i, layer in enumerate(self.layers):
            out = layer(x)
            if self.use_bn:
                out = self.bns[i](out)
            if self.use_ln:
                out = self.lns[i](out)
            out = F.relu(out)
            if self.use_dropout:
                out = self.dropouts[i](out)
            # 处理跳连接逻辑
            if self.use_skip and residual is not None :
                # 判断是否满足 skip_step
                if i % self.skip_step == 0:
                    if self.skip_type == 'normal':
                        # 正常残差连接
                        if residual.shape == out.shape:
                            out = out + residual
                        else:
                            raise ValueError(f"Skip connection shape mismatch: {residual.shape} vs {out.shape}")
                    elif self.skip_type == 'random':
                        if self.training:
                            # 训练阶段，以一定概率跳过残差连接，直接使用恒等映射
                            p = random.random()
                            if  p > self.skip_drop_prob:
                                if residual.shape == out.shape:
                                    out = out + residual
                                else:
                                    raise ValueError(f"Skip connection shape mismatch: {residual.shape} vs {out.shape}")
                            else:
                                out = residual
                        else:
                            # 推理阶段：恒加残差，但是乘以残差保留概率以保持期望不变
                            scale = 1.0 - self.skip_drop_prob
                            if residual.shape == out.shape:
                                out = out + residual * scale
                            else:
                                raise ValueError(f"Skip connection shape mismatch: {residual.shape} vs {out.shape}")

            x = out
            # ===== 更新 residual 供下一轮 skip 使用 =====
            if self.use_skip and i % self.skip_step == 0:
                residual = x
        out = self.output_layer(x)
        return out

if __name__ == '__main__':
    random.seed(42)
    torch.manual_seed(42)
    a = {"a":1, "b":2}
    reg_config = {
        "use_dropout":True,
        "drop_rate": [0.1, 0.2, 0.3],
        "use_bn": True,
        "use_ln": True,
        "use_skip": True,
        "skip_step": 1,
        "skip_type": "random",
        "skip_drop_prob": 0.5
    }
    model = BackboneMLP(
        input_dim=16,
        output_dim=4,
        hidden_dims=[32, 32, 32],
        reg_config=reg_config
    )
    x = torch.randn(4, 16)         # batch_size=4, input_dim=16
    output = model(x)

    print(f"\n✅ Output shape: {output.shape}")