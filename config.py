"""
Configuration module for Transformer model training.

This module defines dataclasses for organizing all configuration
parameters, including dataset paths, model architecture,
training hyperparameters, runtime settings, and checkpoint handling.

It also provides a `LogLevel` enum for setting logging verbosity.
"""
import pprint
import logging
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class RuntimeConfig:
    """
    Runtime execution and debugging settings.

    Attributes:
        logging_level (LogLevel): Logging verbosity during execution.
        count_param_only (bool): If True, only count model parameters and skip training.
        seed (int): Random seed for reproducibility.
        num_workers (int): Number of DataLoader workers for parallel data loading.
    """
    logging_level: "LogLevel" = None        # Logging verbosity (default set in __post_init__)
    count_param_only: bool = False          # Only count parameters, skip training
    seed: int = 73                          # Random seed for reproducibility
    num_workers: int = 4                    # DataLoader parallel worker count

    def __post_init__(self):
        if self.logging_level is None:
            self.logging_level = LogLevel.WARNING

    def set_logging_level(self, level: "LogLevel" = None):
        """
        Set the runtime logging level.

        Args:
            level (LogLevel): Desired logging verbosity.
        """
        if level is None:
            level = self.logging_level or LogLevel.WARNING

        self.logging_level = level
        # Execute the basic configuration
        logging.basicConfig(level=self.logging_level.name,
                            format='%(levelname)s - %(message)s')
        logging.info(f"Logging level set to {level}")


@dataclass
class ModelConfig:
    """Configuration for Transformer model architecture."""
    embed_dim: int = 512                # Embedding dimension
    num_heads: int = 8                  # Number of attention heads
    num_layers: int = 6                 # Number of Encoder/Decoder layers
    d_k: int = 64                       # Dimension for K-space
    d_v: int = 64                       # Dimension for V-space
    d_ff: int = 2048                    # Feed-forward hidden dimension
    dropout: float = 0.1                # Dropout probability


@dataclass
class TrainingConfig:
    """Configuration for model training hyperparameters."""
    batch_size: int = 32                # Batch size
    epochs: int = 40                    # Number of epochs
    max_grad_clip: float = 1.0          # Gradient clipping threshold
    label_smoothing: float = 0.1        # Label smoothing parameter
    learning_rate: float = 1e-5         # Initial learning rate
    betas: tuple[float, float] = (0.9, 0.98)  # Adam optimizer betas
    epsilon: float = 1e-9               # Adam optimizer's epsilon (numerical stability)
    warmup_steps: int = 1000            # Scheduler warmup period (number of steps)
    weight_decay: float = 1e-5          # Weight decay parameter (L2 regularization)
    patience: int = 5                   # Early stopping patience (epochs)
    accumulation_steps: int = 32        # Gradient accumulation steps


class LogLevel(Enum):
    """
    Defines standard logging levels using `logging` module's integer values.

    Provides a string conversion method for cleaner printing and
    a helper to create log levels from string input.
    """
    DEBUG    = logging.DEBUG
    INFO     = logging.INFO
    WARNING  = logging.WARNING
    ERROR    = logging.ERROR
    CRITICAL = logging.CRITICAL

    def __str__(self):
        """Return the name of the log level (e.g., 'DEBUG')."""
        return self.name

    @classmethod
    def from_string(cls, level_str: str) -> "LogLevel":
        """
        Convert a string to a `LogLevel`, case-insensitive.

        Args:
            level_str: Log level name (e.g., 'debug', 'INFO').

        Returns:
            LogLevel: Matching enum member.

        Raises:
            ValueError: If the string does not match any log level.
        """
        try:
            return cls[level_str.upper()]
        except KeyError:
            raise ValueError(
                f"Invalid log level: {level_str}. "
                f"Choose from: {[lvl.name for lvl in cls]}"
            )


@dataclass
class DatasetPaths:
    """
    Stores dataset file paths for both debug and full modes.

    Automatically generates paths for training, validation,
    and test datasets in JSON format based on `base_path`
    and `dataset_name`.
    """
    base_path: str = "data/local_datasets"
    dataset_name: str = "iwslt14"

    # Dictionaries storing paths for debug and full dataset versions
    debug: dict[str, str] = field(init=False)
    full: dict[str, str] = field(init=False)

    def __post_init__(self):
        """Generate debug and full dataset paths after initialization."""
        self.debug = self._make_paths(suffix="_debug")
        self.full = self._make_paths()

    def _make_paths(self, suffix: str = "") -> dict[str, str]:
        """
        Create dataset paths for train/validation/test sets.

        Args:
            suffix: Optional suffix to append to dataset file names.

        Returns:
            dict: Mapping split names to JSON file paths.
        """
        return {
            "train":      f"{self.base_path}/{self.dataset_name}_train{suffix}.json",
            "validation": f"{self.base_path}/{self.dataset_name}_validation{suffix}.json",
            "test":       f"{self.base_path}/{self.dataset_name}_test{suffix}.json"
        }

    def get(self, use_debug: bool = False) -> dict[str, str]:
        """
        Retrieve dataset paths depending on debug mode.

        Args:
            use_debug: If True, returns debug dataset paths. Default is False.

        Returns:
            dict: Selected dataset paths.
        """
        return self.debug if use_debug else self.full


@dataclass
class CheckpointConfig:
    """
    Configuration for saving and loading model checkpoints.

    Allows using a default checkpoint path or setting a custom one.
    """
    checkpoint_path: Path = Path("model_checkpoint.pth")
    _custom_path: Path | None = field(init=False, default=None)

    @property
    def model_path(self) -> Path:
        """
        Returns the final path for the model checkpoint.

        If a custom path has been set, returns it.
        Otherwise, returns the default `checkpoint_path`.
        """
        return self._custom_path if self._custom_path else self.checkpoint_path

    def set_custom_path(self, user_path: str):
        """
        Set a custom checkpoint save path.

        Args:
            user_path: File path as a string.
        """
        self._custom_path = Path(user_path)
        logging.info(f"Custom path for checkpoint set to: {self.model_path}")


@dataclass
class Config:
    """
    Top-level configuration object aggregating all config sections.

    Includes dataset paths, model settings, training hyperparameters,
    runtime configuration, and checkpoint handling.
    """
    dataset_paths: DatasetPaths = field(default_factory=DatasetPaths)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    def __str__(self):
        """Pretty-print configuration as a dictionary."""
        return pprint.pformat(self.__dict__, indent=2)
