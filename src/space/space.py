from ray import tune
import numpy as np

def log_choice(low, high, bins=30):
    log_vals = np.linspace(np.log10(low), np.log10(high), bins)
    # print(10 ** log_vals)
    return float(10 ** np.random.choice(log_vals))

def quniform_choice(low, high, q):
    # 生成候选点（包含 high）
    values = np.arange(low + q, high + q, q)
    # 从离散集合均匀采样
    # print(values)
    return np.random.choice(values)

def category_choice(options):
    return np.random.choice(options)


def get_default_reg():
    config = {
        "use_l1": False,
        "l1_lambda": 0.0,
        "use_l2": False,
        "l2_lambda": 0.0,
        "use_dropout": False,
        "drop_rate": 0.0,
        "use_bn": False,
        "use_ln": False,
        "use_skip": False,
        "skip_type": "None",
        "skip_step": 1,
        "skip_drop_prob": 0.0,
        "use_data_augment": False,
        "da_type": "None",
        "cutout_ratio": 0.0,
        "cutout_prob": 0.0,
        "mixup_alpha": 0.0,
        "mixup_prob": 0.0,
        "cutmix_alpha": 0.0,
        "cutmix_prob": 0.0,
        "fgsm_epsilon": 0.0,
        "fgsm_prob": 0.0,
        "use_swa": False,
        "use_lookahead": False,
    }
    return config


reg_space = {
    "use_l1": tune.choice([True, False]),
    "l1_lambda": tune.sample_from(
        lambda spec: log_choice(1e-6, 1e-2)  if spec["use_l1"] else 0.0
    ),
    "use_l2": tune.choice([True, False]),
    "l2_lambda": tune.sample_from(
        lambda spec: log_choice(1e-6, 1e-2) if spec["use_l2"] else 0.0
    ),
    "use_dropout": tune.choice([True, False]),
    "drop_rate": tune.sample_from(
        lambda spec: quniform_choice(0.0, 0.5, 0.05) if spec["use_dropout"] else 0.0
    ),
    "use_bn": tune.choice([True, False]),
    "use_ln": tune.choice([True, False]),
    "use_skip": tune.choice([True, False]),
    "skip_type": tune.sample_from(
        lambda spec: category_choice(["normal", "random"]) if spec["use_skip"] else "None",
    ),
    "skip_step": tune.sample_from(
        lambda spec: category_choice([1]) if spec["use_skip"] else 1,
    ),
    "skip_drop_prob": tune.sample_from(
        lambda spec: quniform_choice(0.0, 0.5, 0.05) if spec["use_skip"] and spec["skip_type"]=="random" else 0.0
    ),
    "use_data_augment": tune.choice([True, False]),
    "da_type": tune.sample_from(
        lambda spec: category_choice(["cutout", "mixup", "cutmix", "fgsm"]) if spec["use_data_augment"] else "None"
    ),
    "cutout_ratio": tune.sample_from(
        lambda spec: quniform_choice(0.0, 0.5, 0.05) if spec["da_type"] == "cutout" else 0.0
    ),
    "cutout_prob": tune.sample_from(
        lambda spec: quniform_choice(0.0, 0.8, 0.1) if spec["da_type"] == "cutout" else 0.0
    ),
    "mixup_alpha": tune.sample_from(
        lambda spec: quniform_choice(0.0, 1.0, 0.1) if spec["da_type"] == "mixup" else 0.0
    ),
    "mixup_prob": tune.sample_from(
        lambda spec: quniform_choice(0.0, 0.8, 0.1) if spec["da_type"] == "mixup" else 0.0
    ),
    "cutmix_alpha": tune.sample_from(
        lambda spec: quniform_choice(0.0, 1.0, 0.1) if spec["da_type"] == "cutmix" else 0.0
    ),
    "cutmix_prob": tune.sample_from(
        lambda spec: quniform_choice(0.0, 0.8, 0.1) if spec["da_type"] == "cutmix" else 0.0
    ),
    "fgsm_epsilon": tune.sample_from(
        lambda spec: quniform_choice(0.0, 0.3, 0.05) if spec["da_type"] == "fgsm" else 0.0
    ),
    "fgsm_prob": tune.sample_from(
        lambda spec: quniform_choice(0.0, 0.8, 0.1) if spec["da_type"] == "fgsm" else 0.0
    ),
    "use_swa": tune.choice([True, False]),
    "use_lookahead": tune.choice([True, False]),
}


if __name__ == '__main__':
    pass



