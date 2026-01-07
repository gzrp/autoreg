import random
import numpy as np
from typing import Dict, Any

from src.searcher.condition_sampler import ConditionalSampler

class RandomSearcher:
    """
    简单的随机搜索器（Random Search）
    每次独立采样一个新配置，不使用历史信息。
    """

    def __init__(
        self,
        search_space: Dict[str, Any],
        metric: str = "loss",
        mode: str = "min",
        max_retry: int = 10,
        seed: int = 42,
    ):
        self.space = search_space
        self.metric = metric
        self.mode = mode
        self.max_retry = max_retry
        self.seed = seed

        np.random.seed(seed)
        random.seed(seed)

        self.random_sampler = ConditionalSampler(search_space)
        self.visited: Dict[str, bool] = {}
        self._scores: Dict[str, float] = {}

    def _cfg_id(self, cfg: Dict[str, Any]) -> str:
        items = tuple(sorted((k, repr(v)) for k, v in cfg.items()))
        return str(items)

    def suggest(self) -> Dict[str, Any]:
        """返回一个随机的新配置"""
        for _ in range(self.max_retry):
            cfg = self.random_sampler.sample()
            cid = self._cfg_id(cfg)
            if cid not in self.visited:
                self.visited[cid] = True
                return cfg
        # 如果多次都重复，则直接返回一个随机配置（容错）
        return self.random_sampler.sample()

    def on_result(self, config: Dict[str, Any], result: Dict[str, Any]):
        """记录一次实验结果（可选）"""
        cfg_id = self._cfg_id(config)
        score = result[self.metric]
        self._scores[cfg_id] = score
