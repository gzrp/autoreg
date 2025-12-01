
ProfileData = {
    "adult": {
        "name": "adult",
        "t1": 1.3270721716880798,  # 1.3270721716880798
        "t2": 2.9415910243988037,  # 2.9415910243988037
    },
    "ccfraud": {
        "name": "ccfraud",
        "t1": 1.3587890644073486, # 1.3587890644073486
        "t2": 6.568149414857229,  # 6.568149414857229
    },
    "connect": {
        "name": "connect",
        "t1": 1.076902415752411,  # 1.076902415752411
        "t2": 3.749661695957184,  # 3.749661695957184
    },
    "clickpred": {
        "name": "clickpred",
        "t1": 1.2257603826522827,
        "t2": 3.135044470787048,
    },
    "dionis": {
        "name": "dionis",
        "t1": 1.5144846539497376,
        "t2": 5.807422924518585,
    },
    "devnagari": {
        "name": "devnagari",
        "t1": 1.5344846539497377,
        "t2": 5.406149195671081,
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