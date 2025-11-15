from typing import Tuple, Callable
from torch.utils.data import DataLoader

from src.data.dataset.adult import get_adult_dataloader, get_adult_dataloader_sampled
from src.data.dataset.ldpa import get_ldpa_dataloader, get_ldpa_dataloader_sampled
from src.data.meta import get_metadata

Triplet = Tuple[DataLoader, DataLoader, DataLoader]
_DATASET_LOADERS :dict[str, Callable[[str, int], Triplet]] = {
    "adult": get_adult_dataloader,
    "ldpa": get_ldpa_dataloader,
}

_DATASET_SAMPLED_LOADERS :dict[str, Callable[[str, int, float], Triplet]] = {
    "adult": get_adult_dataloader_sampled,
    "ldpa": get_ldpa_dataloader_sampled,
}

def get_dataloader(dataset: str, data_dir: str, batch_size: int) -> Triplet:
    key = dataset.lower()
    try:
        fn = _DATASET_LOADERS[key]
    except KeyError:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Available: {', '.join(sorted(_DATASET_LOADERS))}"
        )
    return fn(data_dir, batch_size)

def get_sampled_dataloader(dataset: str, data_dir: str, batch_size: int, sample_ratio:float = 0.2) -> Triplet:
    key = dataset.lower()
    try:
        fn = _DATASET_SAMPLED_LOADERS[key]
    except KeyError:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Available: {', '.join(sorted(_DATASET_SAMPLED_LOADERS))}"
        )
    return fn(data_dir, batch_size, sample_ratio)


if __name__ == '__main__':
    meta = get_metadata("adult")
    _train_loader, _val_loader, _test_loader = get_dataloader(meta.get("name"), meta.get("data_dir"), meta.get("batch_size"))
    print(f"Train: {len(_train_loader)} batches, Val: {len(_val_loader)}, Test: {len(_test_loader)}")