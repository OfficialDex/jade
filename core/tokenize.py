import json
import time
from pathlib import Path

import numpy as np

default_path = Path("tokenizer/jade.json")
special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]

_tokenizer = None
_mode = None


def train(shard_dir, vocab_size, output_path=default_path, min_frequency=2):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=special_tokens)

    shard_dir = Path(shard_dir)
    manifest_data = json.loads((shard_dir / "manifest.json").read_text())
    files = [str(shard_dir / entry["name"]) for entry in manifest_data["shards"]]

    def text_iterator():
        for file in files:
            for line in Path(file).read_text().splitlines():
                yield json.loads(line)["text"]

    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    return output_path


def load(path=default_path):
    global _tokenizer, _mode
    path = Path(path)
    if path.exists():
        from tokenizers import Tokenizer
        _tokenizer = Tokenizer.from_file(str(path))
        _mode = "jade"
    else:
        import tiktoken
        _tokenizer = tiktoken.get_encoding("gpt2")
        _mode = "bootstrap"
    return _tokenizer


def get():
    if _tokenizer is None:
        load()
    return _tokenizer


def encode(text):
    tokenizer = get()
    if _mode == "jade":
        return tokenizer.encode(text).ids
    return tokenizer.encode_ordinary(text)


def decode(ids):
    return get().decode(ids)


def vocab_size():
    tokenizer = get()
    return tokenizer.get_vocab_size() if _mode == "jade" else tokenizer.n_vocab


def eos_id():
    tokenizer = get()
    return tokenizer.token_to_id("<eos>") if _mode == "jade" else tokenizer.eot_token


def tokenize_shards(shard_dir, output_dir):
    shard_dir = Path(shard_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_data = json.loads((shard_dir / "manifest.json").read_text())
    entries = []
    eos = eos_id()

    for shard_entry in manifest_data["shards"]:
        ids = []
        for line in (shard_dir / shard_entry["name"]).read_text().splitlines():
            ids.extend(encode(json.loads(line)["text"]))
            ids.append(eos)
        array = np.array(ids, dtype=np.uint16)
        name = shard_entry["name"].replace(".jsonl", ".bin")
        array.tofile(output_dir / name)
        entries.append({"name": name, "tokens": len(array)})

    token_manifest = {"mode": _mode, "vocab_size": vocab_size(), "shards": entries, "total_tokens": sum(e["tokens"] for e in entries)}
    (output_dir / "manifest.json").write_text(json.dumps(token_manifest, indent=2))
    return token_manifest


def benchmark(sample_text, iterations=1000):
    start = time.time()
    ids = []
    for _ in range(iterations):
        ids = encode(sample_text)
    elapsed = time.time() - start
    return {
        "chars_per_sec": len(sample_text) * iterations / elapsed,
        "tokens_per_sec": len(ids) * iterations / elapsed,
        "compression_ratio": len(sample_text) / max(len(ids), 1),
    }

