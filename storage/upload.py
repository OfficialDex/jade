from pathlib import Path

from huggingface_hub import HfApi


def push_file(path, repo_id, path_in_repo=None, repo_type="model"):
    api = HfApi()
    api.create_repo(repo_id, repo_type=repo_type, exist_ok=True)
    api.upload_file(path_or_fileobj=str(path), path_in_repo=path_in_repo or Path(path).name, repo_id=repo_id, repo_type=repo_type)


def push_folder(local_dir, repo_id, path_in_repo="", repo_type="dataset"):
    api = HfApi()
    api.create_repo(repo_id, repo_type=repo_type, exist_ok=True)
    api.upload_folder(folder_path=str(local_dir), path_in_repo=path_in_repo, repo_id=repo_id, repo_type=repo_type)

