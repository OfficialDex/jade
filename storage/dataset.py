dfrom storage.cache import dataset_path
from storage.download import pull_snapshot


def restore(repo_id, name, repo_type="dataset"):
    target = dataset_path(name)
    if target.exists() and any(target.iterdir()):
        return target
    pull_snapshot(repo_id, local_dir=str(target), repo_type=repo_type)
    return target
ef not_implemented(*args, **kwargs):
    raise NotImplementedError("scheduled for a later milestone")
