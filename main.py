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
    return parser


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

    logger.warning(f"command '{args.command}' is not implemented yet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
