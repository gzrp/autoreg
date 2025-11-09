from typing import Tuple, Callable
from torch.utils.data import Dataset

from src.data.dataset.adult import get_adult_dataset, get_adult_dataset_sampled
from src.data.meta import get_metadata

Triplet = Tuple[Dataset, Dataset, Dataset]

_DATASET :dict[str, Callable[[str], Triplet]] = {
    "adult": get_adult_dataset,
}

_DATASET_SAMPLED: dict[str, Callable[[str, float], Triplet]] = {
    "adult": get_adult_dataset_sampled
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
    meta = get_metadata("adult")
    train_set, val_set, test_set = get_dataset(meta.get("name"), meta.get("data_dir"))
    print(f"Train: {len(train_set)} batches, Val: {len(val_set)}, Test: {len(test_set)}")
    train_set, val_set, test_set = get_dataset_sampled(meta.get("name"), meta.get("data_dir"), sample_ratio=0.2)
    print(f"Train: {len(train_set)} batches, Val: {len(val_set)}, Test: {len(test_set)}")