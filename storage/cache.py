from pathlib import Path

from configs.system import checkpoint_dir, dataset_dir


def ensure(path):
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def checkpoint_path(name=""):
    return ensure(checkpoint_dir) / name if name else ensure(checkpoint_dir)


def dataset_path(name=""):
    return ensure(dataset_dir) / name if name else ensure(dataset_dir)

