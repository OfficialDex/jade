import hashlib
import json
import time
from pathlib import Path


def normalize(text):
    return " ".join(text.split()).lower()


def content_hash(text):
    return hashlib.sha256(normalize(text).encode()).hexdigest()


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]


def passes_filter(text, filters):
    if not text:
        return False
    length = len(text)
    if length < filters["min_chars"] or length > filters["max_chars"]:
        return False
    alpha = sum(c.isalpha() for c in text)
    if alpha / max(length, 1) < filters["min_alpha_ratio"]:
        return False
    return True


def filter_records(records, filters):
    for record in records:
        if passes_filter(record["text"], filters):
            yield record


def deduplicate(records):
    seen = set()
    for record in records:
        digest = content_hash(record["text"])
        if digest in seen:
            continue
        seen.add(digest)
        record["hash"] = digest
        yield record


def shard(records, shard_dir, shard_size, prefix="shard"):
    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    buffer = []
    index = 0
    entries = []

    def flush():
        nonlocal index
        if not buffer:
            return
        name = f"{prefix}-{index:05d}.jsonl"
        path = shard_dir / name
        with open(path, "w") as f:
            for record in buffer:
                f.write(json.dumps(record) + "\n")
        entries.append({"name": name, "records": len(buffer), "hash": hashlib.sha256(path.read_bytes()).hexdigest()})
        index += 1
        buffer.clear()

    for record in records:
        buffer.append(record)
        if len(buffer) >= shard_size:
            flush()
    flush()

    return entries


def load_source(source, sample_size, seed):
    from datasets import load_dataset

    dataset = load_dataset(source["path"], name=source.get("name"), split=source.get("split", "train"), streaming=True, revision=source.get("revision"))
    dataset = dataset.shuffle(seed=seed, buffer_size=10000)
    count = 0
    for row in dataset:
        text = row.get("text")
        if text:
            yield {"text": text, "source": source["path"]}
            count += 1
        if count >= sample_size:
            break


def build(config, output_dir, records=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if records is None:
        records = []
        for source in config["sources"].values():
            records.extend(load_source(source, config["sample_size"], config["seed"]))

    raw_count = len(records)
    filtered = list(filter_records(records, config["filters"]))
    deduped = list(deduplicate(filtered))
    entries = shard(deduped, output_dir, config["shard_size"])

    manifest_data = {
        "config_hash": config_hash(config),
        "config": config,
        "raw_count": raw_count,
        "filtered_count": len(filtered),
        "deduped_count": len(deduped),
        "shards": entries,
        "created": time.time(),
    }

    (output_dir / "manifest.json").write_text(json.dumps(manifest_data, indent=2))
    return manifest_data

