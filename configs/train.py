tiny = {
    "batch_size": 8,
    "grad_accum": 1,
    "learning_rate": 0.0003,
    "muon_learning_rate": 0.02,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "steps": 200,
    "warmup": 20,
    "checkpoint_every": 50,
    "eval_every": 50,
}

small = {
    "batch_size": 32,
    "grad_accum": 2,
    "learning_rate": 0.0003,
    "muon_learning_rate": 0.02,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "steps": 5000,
    "warmup": 200,
    "checkpoint_every": 500,
    "eval_every": 500,
}

baseline = {
    "batch_size": 64,
    "grad_accum": 4,
    "learning_rate": 0.0002,
    "muon_learning_rate": 0.02,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "steps": 50000,
    "warmup": 1000,
    "checkpoint_every": 1000,
    "eval_every": 1000,
}

