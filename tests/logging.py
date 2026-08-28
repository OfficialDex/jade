import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.logging import setup, get


def test_setup_returns_logger():
    logger = setup(name="jade_test", level="debug")
    assert logger.name == "jade_test"
    assert logger.level == 10


def test_get_returns_same_logger():
    setup(name="jade_test_get")
    logger = get("jade_test_get")
    assert logger.name == "jade_test_get"


def run():
    test_setup_returns_logger()
    test_get_returns_same_logger()
    print("logging tests passed")


if __name__ == "__main__":
    run()
