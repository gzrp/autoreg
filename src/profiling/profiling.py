
ProfileData = {
    "ccfraud": {
        "name": "ccfraud",
        # "t1": 0.457908,  # 0.4579075140953064
        # "t2": 3.246728,  # 3.24672739982605
        "t1": 1.3587890644073486, # 1.3587890644073486
        "t2": 6.568149414857229,  # 6.568149414857229
    }
}

def get_profile_data(dataset: str) -> dict:
    key = dataset.lower()
    try:
        meta = ProfileData[key]
    except KeyError:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Available: {', '.join(sorted(ProfileData))}"
        )
    return meta