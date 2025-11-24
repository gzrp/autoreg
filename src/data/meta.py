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
    "ccfraud": {
        "name": "ccfraud",
        "instance": 284807,
        "feats": 31,
        "in_features": 30,
        "out_features": 2,
        "batch_size": 128,
        "is_balanced": False,
        "class_ratio": [181965, 311],
        "data_dir": "/data/ruipeng/workdir/autoreg/.data/ccfraud"
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
        "data_dir": "/data/ruipeng/workdir/autoreg/.data/connect"
    },
    "clickpred": {
        "name": "clickpred",
        "instance": 399482,
        "feats": 12,
        "in_features": 15,
        "out_features": 2,
        "batch_size": 128,
        "is_balanced": False,
        "class_ratio": [212748, 42920],
        "data_dir": "/data/ruipeng/workdir/autoreg/.data/clickpred"
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
    print(get_metadata(dataset="clickpred"))