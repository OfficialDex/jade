import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import checkpoint
from storage import manifest, download


def fake_state(step):
    return checkpoint.build(
        model_state={"weight": step},
        optimizer_state={"lr": 0.001},
        scheduler_state={"warmup": 10},
        rng_state={"seed": 42},
        step=step,
        epoch=1,
        tokens=step * 1000,
        config={"hidden_size": 128},
        tokenizer_info={"vocab": 8000},
        dataset_info={"name": "tiny"},
        experiment="e999",
        commit="deadbeef",
    )


def test_commit_verifies_and_saves():
    checkpoint_dir = "/tmp/jade_test/ckpt_a"
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    ok, path = checkpoint.commit(fake_state(10), 10, checkpoint_dir)
    assert ok
    assert Path(path).exists()


def test_commit_keeps_latest_and_previous_only():
    checkpoint_dir = "/tmp/jade_test/ckpt_b"
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    for step in [10, 20, 30]:
        ok, _ = checkpoint.commit(fake_state(step), step, checkpoint_dir, keep=2)
        assert ok
    remaining = sorted(Path(checkpoint_dir).glob("checkpoint-*.pt"))
    assert len(remaining) == 2
    assert "checkpoint-30.pt" in [p.name for p in remaining]
    assert "checkpoint-20.pt" in [p.name for p in remaining]
    assert "checkpoint-10.pt" not in [p.name for p in remaining]


def test_restore_local_returns_latest():
    checkpoint_dir = "/tmp/jade_test/ckpt_c"
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    checkpoint.commit(fake_state(5), 5, checkpoint_dir)
    checkpoint.commit(fake_state(15), 15, checkpoint_dir)
    state = checkpoint.restore(checkpoint_dir)
    assert state["step"] == 15


def test_restore_survives_runtime_destruction():
    remote_dir = Path("/tmp/jade_test/remote")
    live_dir = Path("/tmp/jade_test/ckpt_d")
    new_runtime_dir = Path("/tmp/jade_test/ckpt_d_new")
    shutil.rmtree(remote_dir, ignore_errors=True)
    shutil.rmtree(live_dir, ignore_errors=True)
    shutil.rmtree(new_runtime_dir, ignore_errors=True)
    remote_dir.mkdir(parents=True)

    original_push_file = None

    def fake_push_file(path, repo_id, path_in_repo=None, repo_type="model"):
        name = path_in_repo or Path(path).name
        shutil.copyfile(path, remote_dir / name)

    def fake_pull_file(repo_id, filename, repo_type="model", cache_dir=None):
        source = remote_dir / filename
        if not source.exists():
            raise FileNotFoundError(filename)
        return str(source)

    from storage import upload as upload_module
    real_push = upload_module.push_file
    upload_module.push_file = fake_push_file
    real_pull = download.pull_file
    download.pull_file = fake_pull_file

    try:
        checkpoint.commit(fake_state(100), 100, str(live_dir), repo_id="fake/repo")
        checkpoint.commit(fake_state(200), 200, str(live_dir), repo_id="fake/repo")

        shutil.rmtree(live_dir, ignore_errors=True)

        state = checkpoint.restore(str(new_runtime_dir), repo_id="fake/repo")
        assert state is not None
        assert state["step"] == 200
    finally:
        upload_module.push_file = real_push
        download.pull_file = real_pull


def test_restore_falls_back_to_previous_if_latest_corrupt():
    remote_dir = Path("/tmp/jade_test/remote2")
    live_dir = Path("/tmp/jade_test/ckpt_e")
    new_runtime_dir = Path("/tmp/jade_test/ckpt_e_new")
    shutil.rmtree(remote_dir, ignore_errors=True)
    shutil.rmtree(live_dir, ignore_errors=True)
    shutil.rmtree(new_runtime_dir, ignore_errors=True)
    remote_dir.mkdir(parents=True)

    def fake_push_file(path, repo_id, path_in_repo=None, repo_type="model"):
        name = path_in_repo or Path(path).name
        shutil.copyfile(path, remote_dir / name)

    def fake_pull_file(repo_id, filename, repo_type="model", cache_dir=None):
        source = remote_dir / filename
        if not source.exists():
            raise FileNotFoundError(filename)
        return str(source)

    from storage import upload as upload_module
    real_push = upload_module.push_file
    upload_module.push_file = fake_push_file
    real_pull = download.pull_file
    download.pull_file = fake_pull_file

    try:
        checkpoint.commit(fake_state(1), 1, str(live_dir), repo_id="fake/repo")
        checkpoint.commit(fake_state(2), 2, str(live_dir), repo_id="fake/repo")

        (remote_dir / "checkpoint-2.pt").write_bytes(b"corrupted")

        state = checkpoint.restore(str(new_runtime_dir), repo_id="fake/repo")
        assert state is not None
        assert state["step"] == 1
    finally:
        upload_module.push_file = real_push
        download.pull_file = real_pull


def run():
    test_commit_verifies_and_saves()
    test_commit_keeps_latest_and_previous_only()
    test_restore_local_returns_latest()
    test_restore_survives_runtime_destruction()
    test_restore_falls_back_to_previous_if_latest_corrupt()
    print("checkpoint tests passed")


if __name__ == "__main__":
    run()

