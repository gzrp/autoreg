import collections
import random
import time
from typing import Dict, Any, Deque

import numpy as np

from src.searcher.condition_sampler import ConditionalSampler

class AgeEvolutionSearcher:
    def __init__(
        self,
        search_space: Dict[str, Any],
        population_size: int = 50,
        sample_size: int = 5,
        metric: str = "loss",
        mode: str = "min",
        max_retry: int = 30,
        seed: int = 42,
        crossover_rate: float = 0.1,
        mutation_rate: float = 0.7,
    ):
        self.space = search_space
        self.population_size = population_size
        self.sample_size = max(1, min(sample_size, population_size))
        self.max_retry = max_retry
        self.metric = metric
        self.mode = mode
        self.seed = seed
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        np.random.seed(seed)
        random.seed(seed)

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

    def _mutate_new_config(self, parent: Dict[str, Any], max_mutate_point=3) -> Dict[str, Any]:
        for _ in range(self.max_retry):
            child = self.random_sampler.mutate(parent, max_mutate_point)
            cid = self._cfg_id(child)
            if cid not in self.visited:
                self.visited[cid] = True
                return child
        return self._random_new_config()

    def _crossover_new_config(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        # 交叉操作
        for _ in range(self.max_retry):
            child = self.random_sampler.crossover(parent1, parent2)
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
        # 排序评分：min 任务用升序，max 任务用降序
        if self.mode == "max":
            sorted_candidates = sorted(
                candidates,
                key=lambda c: self._scores.get(self._cfg_id(c), float("-inf")),
                reverse=True,
            )
        else:
            sorted_candidates = sorted(
                candidates,
                key=lambda c: self._scores.get(self._cfg_id(c), float("inf")),
                reverse=False,
            )

        parent1 = sorted_candidates[0]
        parent2 = sorted_candidates[1]  # 次优个体
        if random.random() < self.crossover_rate:
            crosser =  self._crossover_new_config(parent1, parent2)
            crosser = self._mutate_new_config(crosser, max_mutate_point=1)
            print(f"crossover: {parent1}- {parent2} -> {crosser}")
            return crosser

        if random.random() < self.mutation_rate:
            if self.mode == "max":
                parent = max(candidates, key=lambda c: self._scores.get(self._cfg_id(c), float("-inf")))
            else:
                parent = min(candidates, key=lambda c: self._scores.get(self._cfg_id(c), float("inf")))
            # print(f"采样时间：{time.time()-start_time}")
            mutater = self._mutate_new_config(parent)
            print(f"mutating: {parent} -> {mutater}")
            return mutater
        randomer = self._random_new_config()
        print(f"random: {randomer}")
        return randomer
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
            # 根据 mode 判断是最小化还是最大化任务
            # if self.mode == "max":
            #     worst = min(self.population, key=lambda c: self._scores.get(self._cfg_id(c), float("-inf")))
            # else:
            #     worst = max(self.population, key=lambda c: self._scores.get(self._cfg_id(c), float("inf")))
            #
            # worst_id = self._cfg_id(worst)
            # self.population.remove(worst)
            # self._scores.pop(worst_id, None)