sources = {
    "fineweb": {
        "path": "HuggingFaceFW/fineweb",
        "name": "sample-10BT",
        "split": "train",
        "revision": "main",
    },
    "commonpile": {
        "path": "common-pile/comma_v0.1_training_dataset",
        "name": None,
        "split": "train",
        "revision": "main",
    },
}

filters = {
    "min_chars": 200,
    "max_chars": 20000,
    "min_alpha_ratio": 0.5,
}

tiny = {
    "sources": sources,
    "filters": filters,
    "sample_size": 2000,
    "shard_size": 500,
    "seed": 42,
}

