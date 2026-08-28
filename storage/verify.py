from pathlib import Path


def exists_and_nonempty(path):
    path = Path(path)
    return path.exists() and path.stat().st_size > 0


def matches_manifest(path, entry):
    return exists_and_nonempty(path) and bool(entry) and entry.get("verified", False)

