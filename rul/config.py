from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency at authoring time
    yaml = None  # type: ignore


@dataclass
class ModelParams:
    n_estimators: int = 500
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    n_jobs: int = -1
    random_state: int = 42


@dataclass
class Config:
    # Data
    data_path: str = str(Path("data/processed/Battery_RUL_processed.csv").as_posix())
    target_col: str = "RUL"
    test_size: float = 0.2
    random_state: int = 42

    # Artifacts
    models_dir: str = str(Path("models").as_posix())
    outputs_dir: str = str(Path("outputs").as_posix())

    # Model hyperparameters
    model: ModelParams = field(default_factory=ModelParams)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "data_path": self.data_path,
            "target_col": self.target_col,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "models_dir": self.models_dir,
            "outputs_dir": self.outputs_dir,
            "model": self.model.__dict__,
        }
        return d


def load_config(path: str | None) -> Config:
    """Load configuration from YAML if available, else return defaults."""
    if path is None:
        return Config()

    cfg_path = Path(path)
    if not cfg_path.exists():
        return Config()

    if yaml is None:
        # YAML not installed; fall back to defaults
        return Config()

    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # Merge with defaults
    default = Config()
    # Override simple fields if present
    for key in [
        "data_path",
        "target_col",
        "test_size",
        "random_state",
        "models_dir",
        "outputs_dir",
    ]:
        if key in raw:
            setattr(default, key, raw[key])

    model_raw = raw.get("model", {})
    if isinstance(model_raw, dict):
        mp = default.model
        for k, v in model_raw.items():
            if hasattr(mp, k):
                setattr(mp, k, v)

    return default
