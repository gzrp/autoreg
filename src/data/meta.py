MetaData = {
    "adult": {
        "name": "adult",
        "instance": 48842,
        "feats": 15,
        "in_features": 106,
        "out_features": 2,
        "batch_size": 128,
        "is_balanced": False,
        "class_ratio": [29724, 9349],
        # "data_dir": "/home/zrp/pycharmProjects/autoreg/.data/adult"
        "data_dir": "/data/ruipeng/workdir/autoreg/.data/adult"
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
