import collections
import copy
import pickle
import random
from typing import Dict, Any, Deque, Tuple, Optional
from ray.tune.search import Searcher

class ConditionalSampler:
    def __init__(self, space: Dict[str, Any]):
        self.space = space
        # 定义所有可以变异的 use_* 模块
        self.use_keys = [
            "use_l1", "use_l2", "use_dropout", "use_bn", "use_ln", "use_skip",
            "use_data_augment", "use_swa", "use_lookahead"
        ]

    @staticmethod
    def _try_sample(domain, spec) -> Tuple[bool, Any]:
        if hasattr(domain, "sampler"):
            v = domain.sample(spec)
            return v
        # 常量
        return domain

    def sample(self) -> Dict[str, Any]:
        cfg = {}
        for k, domain in self.space.items():
            v = self._try_sample(domain, cfg)
            cfg[k] = v
        return cfg

    def mutate(self, parent: Dict[str, Any], max_retry: int = 100) -> Dict[str, Any]:
        child = copy.deepcopy(parent)
        for _ in range(max_retry):
            # 选择一个进行变异
            chosen = random.choice(self.use_keys)
            # 变异选择逻辑
            if chosen == "use_l1":
                # 随机选择 use_l1 的值并更新 l1_lambda
                child["use_l1"] = self.space["use_l1"].sample(child)
                child["l1_lambda"] = self.space["l1_lambda"].sample(child)
            elif chosen == "use_l2":
                child["use_l2"] = self.space["use_l2"].sample(child)
                child["l2_lambda"] = self.space["l2_lambda"].sample(child)
            elif chosen == "use_dropout":
                child["use_dropout"] = self.space["use_dropout"].sample(child)
                child["drop_rate"] = self.space["drop_rate"].sample(child)
            elif chosen == "use_bn":
                child["use_bn"] = self.space["use_bn"].sample(child)
            elif chosen == "use_ln":
                child["use_ln"] = self.space["use_ln"].sample(child)
            elif chosen == "use_skip":
                child["use_skip"] = self.space["use_skip"].sample(child)
                child["skip_type"] = self.space["skip_type"].sample(child)
                child["skip_step"] = self.space["skip_step"].sample(child)
                child["skip_drop_prob"] = self.space["skip_drop_prob"].sample(child)
            elif chosen == "use_data_augment":
                child["use_data_augment"] = self.space["use_data_augment"].sample(child)
                child["da_type"] = self.space["da_type"].sample(child)
                child["cutout_ratio"] = self.space["cutout_ratio"].sample(child)
                child["cutout_prob"] = self.space["cutout_prob"].sample(child)
                child["mixup_alpha"] = self.space["mixup_alpha"].sample(child)
                child["mixup_prob"] = self.space["mixup_prob"].sample(child)
                child["cutmix_alpha"] = self.space["cutmix_alpha"].sample(child)
                child["cutmix_prob"] = self.space["cutmix_prob"].sample(child)
                child["fgsm_epsilon"] = self.space["fgsm_epsilon"].sample(child)
                child["fgsm_prob"] = self.space["fgsm_prob"].sample(child)
            elif chosen == "use_swa":
                child["use_swa"] = self.space["use_swa"].sample(child)
            elif chosen == "use_lookahead":
                child["use_lookahead"] = self.space["use_lookahead"].sample(child)

            # 检查是否变异成功，确保子代与父代不同
            if child != parent:
                return child
        return child


class AgeEvolutionSearcher(Searcher):
    def __init__(
            self,
            search_space: Dict[str, Any],
            population_size: int = 10,
            sample_size: int = 3,
            metric: str = "loss",
            mode: str = "min",
            max_retry: int = 30
    ):
        """
        :param search_space: 探索空间
        :param population_size: 种群人口上限
        :param sample_size: 候选父代数量
        :param metric: 评价指标名
        :param mode: max/min
        :param max_retry: 为找到未访问过的新配置的最大尝试次数
        """
        super().__init__(metric=metric, mode=mode)
        self.space = search_space
        self.population_size = population_size
        self.sample_size = max(1, min(sample_size, population_size))
        self.max_retry = max_retry
        # 种群，按照年龄顺序维护，队首最老，队尾最新
        # 存 # [{"id": str, "config": dict}]
        self.population: Deque[Dict[str, Any]] = collections.deque()
        # 存 [id,...]
        self.population_ids: Deque[str] = collections.deque()
        self.visited: Dict[str, bool] = {}   # 去重
        self._trial_to_config: Dict[str, Dict[str, Any]] = {}
        self._scores: Dict[str, float] = {}  # id -> metric score

        self.random_sampler = ConditionalSampler(search_space)

    def _random_config(self):
        return {k: v.sample() if hasattr(v, "sample") else random.choice(v)
                for k, v in self.space.items()}

    @staticmethod
    def _cfg_id(cfg: Dict[str, Any]) -> str:
        # 用 tuple(sorted) 作为稳定 key，再转 str
        # 若值不可哈希（如列表），转成 repr 以保证稳定字符串
        items = tuple(sorted((k, repr(v)) for k, v in cfg.items()))
        return str(items)

    def _random_new_config(self) -> Optional[Dict[str, Any]]:
        """尝试随机采样一个未访问过的配置，最多 self.max_retry 次。"""
        while True:
            cfg = self.random_sampler.sample()
            cid = self._cfg_id(cfg)
            if cid not in self.visited:
                return cfg

    def _mutate_new_config(self, parent: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """尝试突变得到未访问过的配置，最多 self.max_retry 次。"""
        for _ in range(self.max_retry):
            child = self.random_sampler.mutate(parent, max_retry=self.max_retry)
            cid = self._cfg_id(child)
            if cid not in self.visited:
                return child
        return self._random_new_config()

    def _best_key(self):
        """根据 mode 返回比较基准（max 为正，min 为负）。"""
        if self.mode == "max":
            return lambda cid: self._scores.get(cid, float("-inf"))
        else:
            return lambda cid: -self._scores.get(cid, float("inf"))

    def suggest(self, trial_id: str) -> Optional[Dict]:
        # 阶段1：未满 -> 随机搜索
        if len(self.population) < self.population_size:
            cfg = self._random_new_config()
            cfg_id = self._cfg_id(cfg)
            self.visited[cfg_id] = True
            self._trial_to_config[trial_id] = cfg
            print(f"随机采样{cfg}")
            return cfg

        # 阶段2：已满 -> 锦标赛选择 + 变异 （失败回退随机）
        candidates = random.sample(list(self.population), k=self.sample_size)

        print(f"候选个体及其分数：")
        for candidate in candidates:
            cfg_id = candidate["id"]
            score = self._scores.get(cfg_id, float("inf"))
            print(f"ID: {cfg_id}, 配置: {candidate['config']}, 分数: {score}")

        key_fn = self._best_key()
        parent = max(candidates, key = lambda one: key_fn(one["id"]))
        parent_cfg = parent["config"]
        print(f"选择的父代配置: {parent_cfg}, 分数: {self._scores.get(parent['id'], float('inf'))}")

        child = self._mutate_new_config(parent_cfg)
        child_id = self._cfg_id(child)
        self.visited[child_id] = True
        self._trial_to_config[trial_id] = copy.deepcopy(child)
        # 打印变异过程
        print(f"变异过程：")
        print(f"父代: {parent_cfg}")
        print(f"子代: {child}")
        return child

    def on_trial_complete(
        self, trial_id: str, result: Optional[Dict] = None, error: bool = False
    ) -> None:
        cfg = self._trial_to_config.pop(trial_id, None)
        if cfg is None:
            return

        cfg_id = self._cfg_id(cfg)
        # 失败或无指标 → 释放 visited，允许未来再次探索该配置
        if error or result is None or self.metric not in result:
            self.visited.pop(cfg_id, None)
            return
        score = result[self.metric]
        # 加入新个体（年龄最新在队尾）
        self.population.append({"id": cfg_id, "config": cfg})
        self.population_ids.append(cfg_id)
        self._scores[cfg_id] = score
        # 淘汰最老
        if len(self.population) > self.population_size:
            oldest = self.population.popleft()
            oldest_id = self.population_ids.popleft()
            self._scores.pop(oldest_id, None)
            # 注意：淘汰不释放 visited（避免重复）

    def save(self, checkpoint_path: str):
        with open(checkpoint_path, "wb") as f:
            pickle.dump((self.population, self.population_ids, self._scores, self.visited), f)

    def restore(self, checkpoint_path: str):
        with open(checkpoint_path, "rb") as f:
            self.population, self.population_ids, self._scores, self.visited = pickle.load(f)
