from typing import Tuple, Callable
from torch.utils.data import Dataset

from src.data.dataset.adult import get_adult_dataset, get_adult_dataset_sampled
from src.data.dataset.bank import get_bank_dataset, get_bank_dataset_sampled
from src.data.dataset.ccfraud import get_ccfraud_dataset, get_ccfraud_dataset_sampled
from src.data.dataset.clickpred import get_clickpred_dataset, get_clickpred_dataset_sampled
from src.data.dataset.connect import get_connect_dataset, get_connect_dataset_sampled
from src.data.dataset.dionis import get_dionis_dataset, get_dionis_dataset_sampled
from src.data.dataset.frappe import get_frappe_dataset, get_frappe_dataset_sampled
from src.data.dataset.walking import get_walking_dataset, get_walking_dataset_sampled
from src.data.meta import get_metadata

Triplet = Tuple[Dataset, Dataset, Dataset]

_DATASET :dict[str, Callable[[str], Triplet]] = {
    "adult": get_adult_dataset,
    "ccfraud": get_ccfraud_dataset,
    "connect": get_connect_dataset,
    "clickpred": get_clickpred_dataset,
    "walking": get_walking_dataset,
    "frappe": get_frappe_dataset,
    "dionis": get_dionis_dataset,
    "bank": get_bank_dataset,
}

_DATASET_SAMPLED: dict[str, Callable[[str, float], Triplet]] = {
    "adult": get_adult_dataset_sampled,
    "ccfraud": get_ccfraud_dataset_sampled,
    "connect": get_connect_dataset_sampled,
    "clickpred": get_clickpred_dataset_sampled,
    "walking": get_walking_dataset_sampled,
    "frappe": get_frappe_dataset_sampled,
    "dionis": get_dionis_dataset_sampled,
    "bank": get_bank_dataset_sampled,
}

def get_dataset(dataset: str, data_dir: str) -> Triplet:
    key = dataset.lower()
    try:
        fn = _DATASET[key]
    except KeyError:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Available: {', '.join(sorted(_DATASET))}"
        )
    return fn(data_dir)

def get_dataset_sampled(dataset: str, data_dir: str, sample_ratio: float) -> Triplet:
    key = dataset.lower()
    try:
        fn = _DATASET_SAMPLED[key]
    except KeyError:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Available: {', '.join(sorted(_DATASET_SAMPLED))}"
        )
    return fn(data_dir, sample_ratio)


if __name__ == '__main__':
    meta = get_metadata("dionis")
    train_set, val_set, test_set = get_dataset(meta.get("name"), meta.get("data_dir"))
    print(f"Train: {len(train_set)} batches, Val: {len(val_set)}, Test: {len(test_set)}")
    train_set, val_set, test_set = get_dataset_sampled(meta.get("name"), meta.get("data_dir"), sample_ratio=0.2)
    print(f"Train: {len(train_set)} batches, Val: {len(val_set)}, Test: {len(test_set)}")