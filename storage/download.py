from huggingface_hub import hf_hub_download, snapshot_download


def pull_file(repo_id, filename, repo_type="model", cache_dir=None):
    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type, cache_dir=cache_dir)


def pull_snapshot(repo_id, local_dir=None, repo_type="dataset", cache_dir=None):
    return snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir=local_dir, cache_dir=cache_dir)

