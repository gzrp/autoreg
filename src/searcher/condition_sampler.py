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

    def mutate(self, parent: Dict[str, Any], max_retry: int = 10) -> Dict[str, Any]:
        for _ in range(max_retry):
            child = copy.deepcopy(parent)
            mutated = False
            # 按概率决定变异点数：1、2、3
            mutation_choices = [1, 2]
            mutation_probs = [0.9, 0.1]
            num_mutations = random.choices(mutation_choices, weights=mutation_probs, k=1)[0]
            selected_keys = random.sample(self.use_keys, num_mutations)
            for use_key in selected_keys:
                mutated = True
                if use_key == "use_l1":
                    # 随机选择 use_l1 的值并更新 l1_lambda
                    # # 3️⃣ 在选中 use_skip 时，增加被采样为 True 的概率
                    # if random.random() < 0.75:  # 75% 概率设为 True
                    #     child["use_l1"] = False
                    # else:
                    #     child["use_l1"] = True
                    child["use_l1"] = self.space["use_l1"].sample(child)
                    child["l1_lambda"] = self.space["l1_lambda"].sample(child)
                elif use_key == "use_l2":
                    child["use_l2"] = self.space["use_l2"].sample(child)
                    child["l2_lambda"] = self.space["l2_lambda"].sample(child)
                elif use_key == "use_dropout":
                    # 3️⃣ 在选中 use_skip 时，增加被采样为 True 的概率
                    # if random.random() < 0.75:  # 75% 概率设为 True
                    #     child["use_dropout"] = True
                    # else:
                    #     child["use_dropout"] = False
                    child["use_dropout"] = self.space["use_dropout"].sample(child)
                    child["drop_rate"] = self.space["drop_rate"].sample(child)
                elif use_key == "use_bn":
                    child["use_bn"] = self.space["use_bn"].sample(child)
                elif use_key == "use_ln":
                    child["use_ln"] = self.space["use_ln"].sample(child)
                    if random.random() < 0.75:  # 75% 概率设为 True
                        child["use_ln"] = True
                    else:
                        child["use_ln"] = False
                elif use_key == "use_skip":
                    # 3️⃣ 在选中 use_skip 时，增加被采样为 True 的概率
                    # if random.random() < 0.75:  # 75% 概率设为 True
                    #     child["use_skip"] = True
                    # else:
                    #     child["use_skip"] = False
                    child["use_skip"] = self.space["use_skip"].sample(child)
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

    def crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any], max_retry: int = 10) -> Dict[str, Any]:
        for _ in range(max_retry):
            # 初始化子代为 parent1 的深拷贝
            child = copy.deepcopy(parent1)
            # 随机确定交叉的键数量（1~max_crossover_point）
            # 按概率决定变异点数：1、2、3
            mutation_choices = [1, 2]
            mutation_probs = [0.9, 0.1]
            num_mutations = random.choices(mutation_choices, weights=mutation_probs, k=1)[0]
            selected_keys = random.sample(self.use_keys, num_mutations)

            for use_key in selected_keys:
                # 随机选择一个父代
                # 从 parent2 复制该模块内容
                child[use_key] = parent2[use_key]
                if use_key == "use_l1":
                    child["l1_lambda"] = parent2["l1_lambda"]
                elif use_key == "use_l2":
                    child["l2_lambda"] = parent2["l2_lambda"]
                elif use_key == "use_dropout":
                    child["drop_rate"] = parent2["drop_rate"]
                elif use_key == "use_bn":
                    pass
                elif use_key == "use_ln":
                    pass
                elif use_key == "use_skip":
                    child["skip_type"] = parent2["skip_type"]
                    child["skip_step"] = parent2["skip_step"]
                    child["skip_drop_prob"] = parent2["skip_drop_prob"]
                elif use_key == "use_data_augment":
                    child["da_type"] = parent2["da_type"]
                    child["cutout_ratio"] = parent2["cutout_ratio"]
                    child["cutout_prob"] = parent2["cutout_prob"]
                    child["mixup_alpha"] = parent2["mixup_alpha"]
                    child["mixup_prob"] = parent2["mixup_prob"]
                    child["cutmix_alpha"] = parent2["cutmix_alpha"]
                    child["cutmix_prob"] = parent2["cutmix_prob"]
                    child["fgsm_epsilon"] = parent2["fgsm_epsilon"]
                    child["fgsm_prob"] = parent2["fgsm_prob"]
                elif use_key == "use_swa":
                    pass
                elif use_key == "use_lookahead":
                    pass

            if child != parent1 and child != parent2:
                return child

        # 失败后返回一个随机的配置
        return self.sample()