tiny = {
    "vocab_size": 8000,
    "hidden_size": 128,
    "layers": 4,
    "heads": 4,
    "kv_heads": 4,
    "intermediate_size": 384,
    "context": 256,
    "rope_theta": 10000,
}

small = {
    "vocab_size": 16000,
    "hidden_size": 512,
    "layers": 8,
    "heads": 8,
    "kv_heads": 4,
    "intermediate_size": 1536,
    "context": 1024,
    "rope_theta": 10000,
}

baseline = {
    "vocab_size": 32000,
    "hidden_size": 1024,
    "layers": 16,
    "heads": 16,
    "kv_heads": 8,
    "intermediate_size": 3072,
    "context": 2048,
    "rope_theta": 10000,
}

