from typing import Tuple, Callable
from torch.utils.data import DataLoader

from src.data.dataset.adult import get_adult_dataloader, get_adult_dataloader_sampled
from src.data.dataset.bank import get_bank_dataloader, get_bank_dataloader_sampled
from src.data.dataset.ccfraud import get_ccfraud_dataloader, get_ccfraud_dataloader_sampled
from src.data.dataset.clickpred import get_clickpred_dataloader, get_clickpred_dataloader_sampled
from src.data.dataset.connect import get_connect_dataloader, get_connect_dataloader_sampled
from src.data.dataset.criteo import get_criteo_dataloader, get_criteo_dataloader_sampled
from src.data.dataset.devnagari import get_devnagari_dataloader, get_devnagari_dataloader_sampled
from src.data.dataset.diabetic import get_diabetic_dataloader, get_diabetic_dataloader_sampled
from src.data.dataset.dionis import get_dionis_dataloader, get_dionis_dataloader_sampled
from src.data.dataset.fashion import get_fashion_dataloader, get_fashion_dataloader_sampled
from src.data.dataset.frappe import get_frappe_dataloader, get_frappe_dataloader_sampled
from src.data.dataset.walking import get_walking_dataloader, get_walking_dataloader_sampled
from src.data.meta import get_metadata

Triplet = Tuple[DataLoader, DataLoader, DataLoader]
_DATASET_LOADERS :dict[str, Callable[[str, int], Triplet]] = {
    "adult": get_adult_dataloader,
    "ccfraud": get_ccfraud_dataloader,
    "connect": get_connect_dataloader,
    "clickpred": get_clickpred_dataloader,
    "walking": get_walking_dataloader,
    "frappe": get_frappe_dataloader,
    "bank": get_bank_dataloader,
    "dionis": get_dionis_dataloader,
    "fashion": get_fashion_dataloader,
    "devnagari": get_devnagari_dataloader,
    "diabetic": get_diabetic_dataloader,
    "criteo": get_criteo_dataloader,
}

_DATASET_SAMPLED_LOADERS :dict[str, Callable[[str, int, float], Triplet]] = {
    "adult": get_adult_dataloader_sampled,
    "ccfraud": get_ccfraud_dataloader_sampled,
    "connect": get_connect_dataloader_sampled,
    "clickpred": get_clickpred_dataloader_sampled,
    "walking": get_walking_dataloader_sampled,
    "frappe": get_frappe_dataloader_sampled,
    "bank": get_bank_dataloader_sampled,
    "dionis": get_dionis_dataloader_sampled,
    "fashion": get_fashion_dataloader_sampled,
    "devnagari": get_devnagari_dataloader_sampled,
    "diabetic": get_diabetic_dataloader_sampled,
    "criteo": get_criteo_dataloader_sampled,

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
    meta = get_metadata("ccfraud")
    _train_loader, _val_loader, _test_loader = get_dataloader(meta.get("name"), meta.get("data_dir"), meta.get("batch_size"))
    print(f"Train: {len(_train_loader)} batches, Val: {len(_val_loader)}, Test: {len(_test_loader)}")

    _train_loader, _val_loader, _test_loader = get_sampled_dataloader(meta.get("name"), meta.get("data_dir"), meta.get("batch_size"))
    print(f"Train: {len(_train_loader)} batches, Val: {len(_val_loader)}, Test: {len(_test_loader)}")