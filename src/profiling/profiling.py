
ProfileData = {
    "adult": {
        "name": "adult",           # 7749
        "t1": 1.155334041595459,   # 2888.3351039886475 * 4 / 10000
        "t2": 2.4298615258932115,  # 4859.723051786423 * 4 / (2000 * 4)
    },
    "bank": {
        "name": "bank",
        "t1": 1.122472637653351,  # 1.3270721716880798
        "t2": 3.5524845870335895,  # 2.9415910243988037
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
        "t1": 1.3974764347076416,
        "t2": 4.8179004192352295,  # 4.8179004192352295
    },
    "frappe": {
        "name": "frappe",
        "t1": 2.3732351751327516,
        "t2": 10.934283079624176,
    },
    "diabetic": {
        "name": "diabetic",  # 1.9125211443901062  5.46336268901825
        "t1": 1.9125211443901062,
        "t2": 5.46336268901825,
    },
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