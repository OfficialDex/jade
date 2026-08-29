import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from configs import model as model_configs
from configs import train as train_configs
from core import train


def make_fake_tokens(path, vocab_size, count):
    tokens = np.random.randint(0, vocab_size, size=count, dtype=np.uint16)
    tokens.tofile(path)


def main():
    work_dir = Path("/tmp/jade_smoke")
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True)

    model_config = dict(model_configs.tiny)
    train_config = dict(train_configs.tiny)
    train_config["steps"] = 10
    train_config["warmup"] = 2
    train_config["checkpoint_every"] = 5
    train_config["batch_size"] = 4
    train_config["grad_accum"] = 1

    token_path = work_dir / "tokens.bin"
    make_fake_tokens(token_path, model_config["vocab_size"], count=50000)

    checkpoint_dir = work_dir / "checkpoints"

    print("running 10 smoke-test steps on a tiny model, no real data, just proving the loop works")
    train.train(
        model_config=model_config,
        train_config=train_config,
        token_path=str(token_path),
        checkpoint_dir=str(checkpoint_dir),
        experiment="smoke",
        git_commit="smoke",
        repo_id=None,
    )
    print("smoke test finished without errors")


if __name__ == "__main__":
    main()

