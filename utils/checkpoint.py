import time
from pathlib import Path

import torch

from storage import manifest, upload, download

required_keys = ["model", "optimizer", "scheduler", "rng", "step", "epoch", "tokens", "config", "tokenizer", "dataset", "experiment", "commit", "created"]


def build(model_state, optimizer_state, scheduler_state, rng_state, step, epoch, tokens, config, tokenizer_info, dataset_info, experiment, commit):
    return {
        "model": model_state,
        "optimizer": optimizer_state,
        "scheduler": scheduler_state,
        "rng": rng_state,
        "step": step,
        "epoch": epoch,
        "tokens": tokens,
        "config": config,
        "tokenizer": tokenizer_info,
        "dataset": dataset_info,
        "experiment": experiment,
        "commit": commit,
        "created": time.time(),
    }


def save(state, path):
    torch.save(state, path)
    return path


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def validate(state):
    return isinstance(state, dict) and all(key in state for key in required_keys)


def verify_file(path):
    try:
        return validate(load(path))
    except Exception:
        return False


def commit(state, step, checkpoint_dir, keep=2, repo_id=None, repo_type="model", log_path=None):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = checkpoint_dir / "manifest.json"

    name = f"checkpoint-{step}.pt"
    path = checkpoint_dir / name
    save(state, path)
    ok = verify_file(path)

    data = manifest.load(manifest_path)
    data = manifest.add(data, name, step, ok, path)

    if ok:
        data, keep_names = manifest.prune(data, keep=keep)
        for entry in checkpoint_dir.glob("checkpoint-*.pt"):
            if entry.name not in keep_names:
                entry.unlink()

    manifest.save(data, manifest_path)

    if ok and repo_id:
        upload.push_file(path, repo_id, name, repo_type)
        upload.push_file(manifest_path, repo_id, "manifest.json", repo_type)
        if log_path and Path(log_path).exists():
            upload.push_file(log_path, repo_id, "jade.log", repo_type)

    return ok, path


def restore(checkpoint_dir, repo_id=None, repo_type="model"):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = checkpoint_dir / "manifest.json"

    if repo_id:
        try:
            remote_path = download.pull_file(repo_id, "manifest.json", repo_type)
            manifest_path.write_text(Path(remote_path).read_text())
        except Exception:
            pass

    data = manifest.load(manifest_path)
    candidates = [manifest.latest(data), manifest.previous(data)]

    for entry in candidates:
        if not entry:
            continue
        local_path = checkpoint_dir / entry["name"]
        if not local_path.exists() and repo_id:
            try:
                remote_path = download.pull_file(repo_id, entry["name"], repo_type)
                local_path.write_bytes(Path(remote_path).read_bytes())
            except Exception:
                continue
        if local_path.exists() and verify_file(local_path):
            return load(local_path)

    return None

