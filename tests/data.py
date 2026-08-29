import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import data


def synthetic_records():
    records = []
    for i in range(20):
        records.append({"text": (f"this is a reasonably long synthetic training example number {i} ") * 10})
    records.append({"text": "short"})
    records.append(dict(records[0]))
    return records


def sample_config():
    return {
        "sources": {},
        "filters": {"min_chars": 50, "max_chars": 5000, "min_alpha_ratio": 0.3},
        "sample_size": 0,
        "shard_size": 5,
        "seed": 42,
    }


def test_filter_removes_short_records():
    filtered = list(data.filter_records(synthetic_records(), sample_config()["filters"]))
    assert all(len(r["text"]) >= 50 for r in filtered)


def test_dedup_removes_exact_duplicate():
    records = synthetic_records()
    filtered = list(data.filter_records(records, sample_config()["filters"]))
    deduped = list(data.deduplicate(filtered))
    hashes = [r["hash"] for r in deduped]
    assert len(hashes) == len(set(hashes))
    assert len(deduped) < len(filtered)


def test_build_is_reproducible():
    output_a = "/tmp/jade_test/data_a"
    output_b = "/tmp/jade_test/data_b"
    shutil.rmtree(output_a, ignore_errors=True)
    shutil.rmtree(output_b, ignore_errors=True)

    config = sample_config()
    manifest_a = data.build(config, output_a, records=synthetic_records())
    manifest_b = data.build(config, output_b, records=synthetic_records())

    assert manifest_a["config_hash"] == manifest_b["config_hash"]
    assert manifest_a["deduped_count"] == manifest_b["deduped_count"]
    assert [e["hash"] for e in manifest_a["shards"]] == [e["hash"] for e in manifest_b["shards"]]


def test_shard_files_written():
    output_dir = "/tmp/jade_test/data_c"
    shutil.rmtree(output_dir, ignore_errors=True)
    manifest_data = data.build(sample_config(), output_dir, records=synthetic_records())
    shard_files = sorted(Path(output_dir).glob("shard-*.jsonl"))
    assert len(shard_files) == len(manifest_data["shards"])
    assert (Path(output_dir) / "manifest.json").exists()


def run():
    test_filter_removes_short_records()
    test_dedup_removes_exact_duplicate()
    test_build_is_reproducible()
    test_shard_files_written()
    print("data tests passed")


if __name__ == "__main__":
    run()

