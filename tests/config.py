import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import config, load_config


def test_defaults():
    c = config()
    assert c.name == "jade"
    assert c.seed == 42


def test_set_get():
    c = config()
    c.set("hidden_size", 128)
    assert c.get("hidden_size") == 128
    assert c.get("missing", "fallback") == "fallback"


def test_save_load(tmp_path="/tmp/jade_test_config.json"):
    c = config(experiment="e001")
    c.set("layers", 4)
    c.save(tmp_path)
    restored = config.load(tmp_path)
    assert restored.experiment == "e001"
    assert restored.get("layers") == 4


def test_load_config_without_file():
    c = load_config("/tmp/does_not_exist.json")
    assert c.name == "jade"


def run():
    test_defaults()
    test_set_get()
    test_save_load()
    test_load_config_without_file()
    print("config tests passed")


if __name__ == "__main__":
    run()
