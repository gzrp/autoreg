import copy
import random
from typing import Dict, Any, Tuple


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

