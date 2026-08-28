import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class config:
    name: str = "jade"
    seed: int = 42
    experiment: str = "e000"
    git_commit: str = ""
    values: dict = field(default_factory=dict)

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, value):
        self.values[key] = value
        return self

    def merge(self, other):
        self.values.update(other)
        return self

    def to_dict(self):
        return {"name": self.name, "seed": self.seed, "experiment": self.experiment, "git_commit": self.git_commit, "values": self.values}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @staticmethod
    def load(path):
        data = json.loads(Path(path).read_text())
        return config(name=data.get("name", "jade"), seed=data.get("seed", 42), experiment=data.get("experiment", "e000"), git_commit=data.get("git_commit", ""), values=data.get("values", {}))

    @staticmethod
    def from_env(prefix="jade_"):
        values = {}
        for key, value in os.environ.items():
            if key.lower().startswith(prefix):
                values[key.lower()[len(prefix):]] = value
        return config(values=values)


def load_env(path=".env"):
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def load_config(path=None):
    load_env()
    base = config()
    if path and Path(path).exists():
        base = config.load(path)
    base.merge(config.from_env().values)
    return base
