from pathlib import Path
from typing import Any, Union
import yaml


class PathManager:
    def __init__(self, config_path: Union[str, Path] = "configs/paths.yaml"):
        self.config_path = Path(config_path).resolve()
        if not self.config_path.exists():
            raise FileNotFoundError(f"Path config not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        if not isinstance(self.cfg, dict):
            raise ValueError(f"Invalid YAML structure in: {self.config_path}")

        # resolve project_root relative to the project where the config file is located
        raw_root = self.cfg.get("project_root", ".")
        raw_root = Path(raw_root)
        if raw_root.is_absolute():
            self.project_root = raw_root.resolve()
        else:
            self.project_root = (self.config_path.parent.parent / raw_root).resolve()

    def _query(self, key: str) -> Any:
        value = self.cfg
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                raise KeyError(f"Path key not found: {key}")
            value = value[part]
        return value

    def _resolve(self, value: Union[str, Path]) -> Path:
        p = Path(value)
        if p.is_absolute():
            return p.resolve()
        return (self.project_root / p).resolve()

    def get_path(self, key: str) -> Path:
        value = self._query(key)
        if not isinstance(value, (str, Path)):
            raise TypeError(f"Path value for '{key}' must be str or Path, got {type(value)}")
        return self._resolve(value)

    # file existence check
    def get_file(self, key: str) -> Path:
        p = self.get_path(key)
        if not p.exists():
            raise FileNotFoundError(f"File not found for '{key}': {p}")
        if not p.is_file():
            raise FileNotFoundError(f"Not a file for '{key}': {p}")
        return p

    def get_dir(self, key: str, create: bool = False) -> Path:
        p = self.get_path(key)
        if create:
            p.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            raise FileNotFoundError(f"Directory not found for '{key}': {p}")
        if not p.is_dir():
            raise NotADirectoryError(f"Not a directory for '{key}': {p}")
        return p


# global instance for direct access
_pm = PathManager()


def get_path(key: str) -> Path:
    return _pm.get_path(key)


def get_file(key: str) -> Path:
    return _pm.get_file(key)


def get_dir(key: str, create: bool = False) -> Path:
    return _pm.get_dir(key, create=create)
