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

        self.key_weights = {
        "use_l1": 3.0,
        "use_l2": 3.0,
        "use_dropout": 3.0,
        "use_bn": 3.0,
        "use_ln": 1.0,
        "use_skip": 3.0,  # 变异时更容易选中
        "use_data_augment": 3.0,
        "use_swa": 1.0,
        "use_lookahead": 1.0,
    }

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

    def mutate(self, parent: Dict[str, Any], max_mutate_point=3, max_retry: int = 30) -> Dict[str, Any]:
        for _ in range(max_retry):
            child = copy.deepcopy(parent)
            mutated = False

            # 随机选取若干个需要变异的模块
            num_mutations = random.randint(1, max_mutate_point)
            selected_keys = random.sample(self.use_keys, num_mutations)
            for use_key in selected_keys:
                mutated = True
                # 选择一个进行变异
                # chosen = random.choices(keys, weights=weights, k=1)[0]           # 变异选择逻辑
                if use_key == "use_l1":
                    # 随机选择 use_l1 的值并更新 l1_lambda
                    # 3️⃣ 在选中 use_skip 时，增加被采样为 True 的概率
                    if random.random() < 0.75:  # 75% 概率设为 True
                        child["use_l1"] = False
                    else:
                        child["use_l1"] = True
                    # child["use_l1"] = self.space["use_l1"].sample(child)
                    child["l1_lambda"] = self.space["l1_lambda"].sample(child)
                elif use_key == "use_l2":
                    child["use_l2"] = self.space["use_l2"].sample(child)
                    child["l2_lambda"] = self.space["l2_lambda"].sample(child)
                elif use_key == "use_dropout":
                    # 3️⃣ 在选中 use_skip 时，增加被采样为 True 的概率
                    if random.random() < 0.75:  # 75% 概率设为 True
                        child["use_dropout"] = True
                    else:
                        child["use_dropout"] = False
                    # child["use_dropout"] = self.space["use_dropout"].sample(child)
                    child["drop_rate"] = self.space["drop_rate"].sample(child)
                elif use_key == "use_bn":
                    child["use_bn"] = self.space["use_bn"].sample(child)
                elif use_key == "use_ln":
                    child["use_ln"] = self.space["use_ln"].sample(child)
                elif use_key == "use_skip":
                    # 3️⃣ 在选中 use_skip 时，增加被采样为 True 的概率
                    if random.random() < 0.75:  # 75% 概率设为 True
                        child["use_skip"] = True
                    else:
                        child["use_skip"] = False
                    # child["use_skip"] = self.space["use_skip"].sample(child)
                    child["skip_type"] = self.space["skip_type"].sample(child)
                    child["skip_step"] = self.space["skip_step"].sample(child)
                    child["skip_drop_prob"] = self.space["skip_drop_prob"].sample(child)
                elif use_key == "use_data_augment":
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
                elif use_key == "use_swa":
                    child["use_swa"] = self.space["use_swa"].sample(child)
                elif use_key == "use_lookahead":
                    child["use_lookahead"] = self.space["use_lookahead"].sample(child)

            if mutated and child != parent:
                return child

        return self.sample()

    def crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any], max_retry: int = 30) -> Dict[str, Any]:
        for _ in range(max_retry):
            # 从两个父代中生成一个新的个体
            child = {}
            for use_key in self.use_keys:
                # 随机选择一个父代
                source = random.choice([parent1, parent2])
                child[use_key] = source[use_key]
                if use_key == "use_l1":
                    child["l1_lambda"] = source["l1_lambda"]
                elif use_key == "use_l2":
                    child["l2_lambda"] = source["l2_lambda"]
                elif use_key == "use_dropout":
                    child["drop_rate"] = source["drop_rate"]
                elif use_key == "use_bn":
                    pass
                elif use_key == "use_ln":
                    pass
                elif use_key == "use_skip":
                    child["skip_type"] = source["skip_type"]
                    child["skip_step"] = source["skip_step"]
                    child["skip_drop_prob"] = source["skip_drop_prob"]
                elif use_key == "use_data_augment":
                    child["da_type"] = source["da_type"]
                    child["cutout_ratio"] = source["cutout_ratio"]
                    child["cutout_prob"] = source["cutout_prob"]
                    child["mixup_alpha"] = source["mixup_alpha"]
                    child["mixup_prob"] = source["mixup_prob"]
                    child["cutmix_alpha"] = source["cutmix_alpha"]
                    child["cutmix_prob"] = source["cutmix_prob"]
                    child["fgsm_epsilon"] = source["fgsm_epsilon"]
                    child["fgsm_prob"] = source["fgsm_prob"]
                elif use_key == "use_swa":
                    pass
                elif use_key == "use_lookahead":
                    pass

            if child != parent1 and child != parent2:
                return child

        # 失败后返回一个随机的配置
        return self.sample()