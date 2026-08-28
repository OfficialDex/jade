tiny = {
    "batch_size": 8,
    "learning_rate": 0.0003,
    "steps": 200,
    "warmup": 20,
    "checkpoint_every": 50,
    "eval_every": 50,
}

small = {
    "batch_size": 32,
    "learning_rate": 0.0003,
    "steps": 5000,
    "warmup": 200,
    "checkpoint_every": 500,
    "eval_every": 500,
}

baseline = {
    "batch_size": 64,
    "learning_rate": 0.0002,
    "steps": 50000,
    "warmup": 1000,
    "checkpoint_every": 1000,
    "eval_every": 1000,
}
