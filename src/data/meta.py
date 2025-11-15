MetaData = {
    "adult": {
        "name": "adult",
        "instance": 48842,
        "feats": 15,
        "in_features": 106,
        "out_features": 2,
        "batch_size": 64,
        "is_balanced": False,
        "class_ratio": [29724, 9349],
        # "data_dir": "/home/zrp/pycharmProjects/autoreg/.data/adult"
        "data_dir": "/data/ruipeng/workdir/autoreg/.data/adult"
    },
    "ldpa": {
        "name": "ldpa",
        "instance": 164860,
        "feats": 8,
        "in_features": 14,
        "out_features": 11,
        "batch_size": 128,
        "is_balanced": False,
        "class_ratio": [2378, 2278, 26168, 43584, 4935, 4168, 21795, 1365, 9423, 14689, 1105],
        # "data_dir": "/home/zrp/pycharmProjects/autoreg/.data/adult"
        "data_dir": "/data/ruipeng/workdir/autoreg/.data/ldpa"
    }
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
