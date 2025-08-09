import pprint
import logging
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field


class LogLevel(Enum):
    """Defines standard logging levels using logging module's integer values."""
    DEBUG    = logging.DEBUG
    INFO     = logging.INFO
    WARNING  = logging.WARNING
    ERROR    = logging.ERROR
    CRITICAL = logging.CRITICAL

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, level_str: str) -> "LogLevel":
        try:
            return cls[level_str.upper()]
        except KeyError:
            raise ValueError(f"Invalid log level: {level_str}. "
                             f"Choose from: {[lvl.name for lvl in cls]}")


@dataclass
class DatasetPaths:
    base_path: str = "data/local_datasets"
    dataset_name: str = "iwslt14"

    debug: dict[str, str] = field(init=False)
    full: dict[str, str] = field(init=False)

    def __post_init__(self):
        self.debug = self._make_paths(suffix="_debug")
        self.full = self._make_paths()

    def _make_paths(self, suffix: str = "") -> dict[str, str]:
        return {
            "train":      f"{self.base_path}/{self.dataset_name}_train{suffix}.json",
            "validation": f"{self.base_path}/{self.dataset_name}_validation{suffix}.json",
            "test":       f"{self.base_path}/{self.dataset_name}_test{suffix}.json"
        }

    def get(self, use_debug: bool) -> dict[str, str]:
        return self.debug if use_debug else self.full


@dataclass
class RuntimeConfig:
    data_debug_mode: bool = True
    logging_level: LogLevel = LogLevel.WARNING
    count_param_only: bool = False


@dataclass
class ModelConfig:
    embed_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    d_k: int = 64
    d_v: int = 64
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 10
    max_grad_clip: float = 1.0
    label_smoothing: float = 0.1
    learning_rate: float = 1e-5
    betas: tuple[float, float] = (0.9, 0.98)
    epsilon: float = 1e-9
    warmup_steps: int = 100
    weight_decay: float = 1.0


@dataclass
class CheckpointConfig:
    checkpoint_path: Path = Path("model_checkpoint.pth")
    _custom_path: Path | None = field(init=False, default=None)

    @property
    def model_path(self) -> Path:
        """Returns the final path for the model checkpoint.

        This will be the user-provided path if set, otherwise it defaults
        to the checkpoint_path.
        """
        # The single source of truth for the model path
        return self._custom_path if self._custom_path else self.checkpoint_path

    def set_custom_path(self, user_path: str):
        """Sets the custom path for the model checkpoint from a string."""
        self._custom_path = Path(user_path)
        logging.info(f"Custom path for checkpoint set to: {self.model_path}")


@dataclass
class Config:
    dataset_paths: DatasetPaths = field(default_factory=DatasetPaths)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    def __str__(self):
        return pprint.pformat(self.__dict__, indent=2)