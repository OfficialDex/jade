import argparse
import subprocess
import sys

from utils.config import load_config
from utils.logging import setup


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def build_parser():
    parser = argparse.ArgumentParser(prog="jade")
    parser.add_argument("command", nargs="?", default="status", choices=["status", "train", "evaluate", "dashboard", "infer"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--experiment", default="e000")
    parser.add_argument("--size", default="tiny", choices=["tiny", "small", "baseline"])
    parser.add_argument("--tokens", default="data/tokens")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--repo-id", default=None)
    return parser


def run_train(args, commit):
    from configs import model as model_configs
    from configs import train as train_configs
    from core import train

    model_config = getattr(model_configs, args.size)
    train_config = getattr(train_configs, args.size)

    train.train(
        model_config=model_config,
        train_config=train_config,
        token_path=args.tokens,
        checkpoint_dir=args.checkpoint_dir,
        experiment=args.experiment,
        git_commit=commit,
        repo_id=args.repo_id,
        log_path="logs/jade.log",
    )


def main():
    args = build_parser().parse_args()
    logger = setup(path="logs/jade.log")
    config = load_config(args.config)
    config.experiment = args.experiment
    config.git_commit = git_commit()

    logger.info(f"jade starting  command={args.command}  experiment={config.experiment}  commit={config.git_commit}")

    if args.command == "status":
        logger.info("environment ok, no active training")
        return 0

    if args.command == "train":
        run_train(args, config.git_commit)
        return 0

    logger.warning(f"command '{args.command}' is not implemented yet")
    return 0


if __name__ == "__main__":
    sys.exit(main())

