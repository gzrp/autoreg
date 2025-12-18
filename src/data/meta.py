MetaData = {
    "adult": {
        "name": "adult",
        "instance": 48842,
        "feats": 15,
        "in_features": 106,
        "out_features": 2,
        "batch_size": 64,
        "is_balanced": True,
        "class_ratio": [29724, 9349],
        "data_dir": "/data/ruipeng/workdir/autoreg/.data/adult",
        "max_epochs": 8,
    },
    "bank": {
        "name": "bank",
        "instance": 45211,
        "feats": 21,
        "in_features": 54,
        "out_features": 2,
        "batch_size": 64,
        "is_balanced": True,
        "class_ratio": [23411, 2949],
        "data_dir": "/data/ruipeng/workdir/autoreg/.data/bank",
        "max_epochs": 16,
    },
    "ccfraud": {
        "name": "ccfraud",
        "instance": 284807,
        "feats": 31,
        "in_features": 30,
        "out_features": 2,
        "batch_size": 256,
        "is_balanced": True,
        "class_ratio": [181965, 311],
        "data_dir": "/data/ruipeng/workdir/autoreg/.data/ccfraud",
        "max_epochs": 16,
    },
    "clickpred": {
        "name": "clickpred",
        "instance": 399482,
        "feats": 12,
        "in_features": 15,
        "out_features": 2,
        "batch_size": 256,
        "is_balanced": True,
        "class_ratio": [212748, 42920],
        "data_dir": "/data/ruipeng/workdir/autoreg/.data/clickpred",
        "max_epochs": 32,
    },
    "diabetic": {
        "name": "diabetic",
        "instance": 101766,
        "feats": 45,
        "in_features": 2347,
        "out_features": 2,
        "batch_size": 64,
        "is_balanced": True,
        "class_ratio": [90409, 11357],            # >30 && No 是 0， <30 是 1,
        "data_dir": "/data/ruipeng/workdir/autoreg/.data/diabetic",
        "max_epochs": 16,
    },
    "frappe": {
        "name": "frappe",
        "instance": 288609,
        "feats": 11,
        "in_features": 5382,
        "out_features": 2,
        "batch_size": 256,
        "is_balanced": True,
        "class_ratio": [134423, 67604],
        "data_dir": "/data/ruipeng/workdir/autoreg/.data/frappe",
        "max_epochs": 16,
    },
    "connect": {
        "name": "connect",
        "instance": 67557,
        "feats": 43,
        "in_features": 126,
        "out_features": 3,
        "batch_size": 64,
        "is_balanced": False,
        "class_ratio": [10650, 28439, 4147],
        "data_dir": "/data/ruipeng/workdir/autoreg/.data/connect",
        "max_epochs": 16,
    },
    "devnagari": {
        "name": "devnagari",
        "instance": 92000,
        "feats": 1025,
        "in_features": 1024,
        "out_features": 46,
        "batch_size": 64,
        "is_balanced": True,
        "class_ratio": None,
        "data_dir": "/data/ruipeng/workdir/autoreg/.data/devnagari",
        "max_epochs": 16,
    },
    "dionis": {
        "name": "dionis",
        "instance": 416188,
        "feats": 61,
        "in_features": 60,
        "out_features": 355,
        "batch_size": 256,
        "is_balanced": True,
        "class_ratio": None,
        "data_dir": "/data/ruipeng/workdir/autoreg/.data/dionis",
        "max_epochs": 32,
    },
}


def get_metadata(dataset: str) -> dict:
    key = dataset.lower()
    try:
        meta = MetaData[key]
    except KeyError:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Available: {', '.join(sorted(MetaData))}"
        )
    return meta


if __name__ == '__main__':
    print(get_metadata(dataset="criteo"))