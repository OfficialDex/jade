import time
import traceback
from contextlib import contextmanager

from utils.logging import get


@contextmanager
def timer(label):
    logger = get()
    start = time.time()
    try:
        yield
    finally:
        logger.debug(f"{label} took {time.time() - start:.3f}s")


@contextmanager
def guard(label):
    logger = get()
    try:
        yield
    except Exception:
        logger.error(f"{label} failed\n{traceback.format_exc()}")
        raise


def assert_state(condition, message):
    if not condition:
        get().error(f"state check failed: {message}")
        raise AssertionError(message)


def snapshot(obj, keys):
    return {key: getattr(obj, key, None) for key in keys}
