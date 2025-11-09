import collections
import random
import time
from typing import Dict, Any, Deque

from src.searcher.condition_sampler import ConditionalSampler

class AgeEvolutionSearcher:
    def __init__(
        self,
        search_space: Dict[str, Any],
        population_size: int = 10,
        sample_size: int = 3,
        metric: str = "loss",
        mode: str = "min",
        max_retry: int = 30
    ):
        self.space = search_space
        self.population_size = population_size
        self.sample_size = max(1, min(sample_size, population_size))
        self.max_retry = max_retry
        self.metric = metric
        self.mode = mode

        self.population: Deque[Dict[str, Any]] = collections.deque()
        self._scores: Dict[str, float] = {}
        self.visited: Dict[str, bool] = {}

        self.random_sampler = ConditionalSampler(search_space)

    def _cfg_id(self, cfg: Dict[str, Any]) -> str:
        items = tuple(sorted((k, repr(v)) for k, v in cfg.items()))
        return str(items)

    def _random_new_config(self) -> Dict[str, Any]:
        """返回一个未访问过的新配置"""
        for _ in range(self.max_retry):
            cfg = self.random_sampler.sample()
            cid = self._cfg_id(cfg)
            if cid not in self.visited:
                self.visited[cid] = True
                return cfg
        # 实在找不到就随机返回一个
        return self.random_sampler.sample()

    def _mutate_new_config(self, parent: Dict[str, Any]) -> Dict[str, Any]:
        for _ in range(self.max_retry):
            child = self.random_sampler.mutate(parent)
            cid = self._cfg_id(child)
            if cid not in self.visited:
                self.visited[cid] = True
                return child
        return self._random_new_config()

    def suggest(self) -> Dict[str, Any]:
        """生成下一个配置"""
        # 若种群未满，随机采样
        start_time = time.time()
        if len(self.population) < self.population_size:
            # print(f"采样时间：{time.time() - start_time}")
            return self._random_new_config()
        # 否则执行“锦标赛选择 + 变异”
        candidates = random.sample(list(self.population), k=self.sample_size)
        if self.mode == "max":
            parent = max(candidates, key=lambda c: self._scores.get(self._cfg_id(c), float("-inf")))
        else:
            parent = min(candidates, key=lambda c: self._scores.get(self._cfg_id(c), float("inf")))

        # print(f"采样时间：{time.time()-start_time}")
        return self._mutate_new_config(parent)
        # return get_default_reg()
    def on_result(self, config: Dict[str, Any], result: Dict[str, Any]):
        """在一个 trial 完成后更新种群"""
        cfg_id = self._cfg_id(config)
        score = result[self.metric]

        self.population.append(config)
        self._scores[cfg_id] = score
        # 淘汰最老个体
        if len(self.population) > self.population_size:
            oldest = self.population.popleft()
            oldest_id = self._cfg_id(oldest)
            self._scores.pop(oldest_id, None)