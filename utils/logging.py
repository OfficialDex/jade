import logging as pylog
import sys
from pathlib import Path


def setup(name="jade", level="info", path=None):
    logger = pylog.getLogger(name)
    logger.setLevel(getattr(pylog, level.upper(), pylog.INFO))
    logger.handlers.clear()

    formatter = pylog.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")

    console = pylog.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        file_handler = pylog.FileHandler(path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def get(name="jade"):
    return pylog.getLogger(name)
