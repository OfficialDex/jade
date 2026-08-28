import json
import time
from pathlib import Path


def empty():
    return {"checkpoints": []}


def load(path):
    path = Path(path)
    if not path.exists():
        return empty()
    return json.loads(path.read_text())


def save(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def add(data, name, step, verified, path):
    data["checkpoints"].append({"name": name, "step": step, "verified": verified, "path": str(path), "created": time.time()})
    return data


def valid_entries(data):
    return [entry for entry in data["checkpoints"] if entry.get("verified")]


def latest(data):
    entries = valid_entries(data)
    return max(entries, key=lambda entry: entry["step"]) if entries else None


def previous(data):
    entries = sorted(valid_entries(data), key=lambda entry: entry["step"], reverse=True)
    return entries[1] if len(entries) > 1 else None


def prune(data, keep=2):
    entries = sorted(valid_entries(data), key=lambda entry: entry["step"], reverse=True)
    keep_names = {entry["name"] for entry in entries[:keep]}
    data["checkpoints"] = [entry for entry in data["checkpoints"] if entry["name"] in keep_names or not entry.get("verified")]
    return data, keep_names

