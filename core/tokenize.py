import json
from pathlib import Path

import numpy as np

_encoding = None


def get():
    global _encoding
    if _encoding is None:
        import tiktoken
        _encoding = tiktoken.get_encoding("gpt2")
    return _encoding


def encode(text):
    return get().encode_ordinary(text)


def decode(ids):
    return get().decode(ids)


def vocab_size():
    return get().n_vocab


def tokenize_shards(shard_dir, output_dir):
    shard_dir = Path(shard_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_data = json.loads((shard_dir / "manifest.json").read_text())
    entries = []

    for shard_entry in manifest_data["shards"]:
        ids = []
        for line in (shard_dir / shard_entry["name"]).read_text().splitlines():
            record = json.loads(line)
            ids.extend(encode(record["text"]))
            ids.append(get().eot_token)
        array = np.array(ids, dtype=np.uint16)
        name = shard_entry["name"].replace(".jsonl", ".bin")
        array.tofile(output_dir / name)
        entries.append({"name": name, "tokens": len(array)})

    token_manifest = {"tokenizer": "gpt2-bootstrap", "vocab_size": vocab_size(), "shards": entries, "total_tokens": sum(e["tokens"] for e in entries)}
    (output_dir / "manifest.json").write_text(json.dumps(token_manifest, indent=2))
    return token_manifest

