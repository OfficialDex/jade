import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests import config, logging, checkpoint


def main():
    config.run()
    logging.run()
    checkpoint.run()
    print("all tests passed")


if __name__ == "__main__":
    main()

