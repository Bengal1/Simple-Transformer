"""
This module provides utility functions and classes for machine learning tasks,
including model checkpoint saving/loading, learning rate scheduling, loss plotting,
and dataset preprocessing for machine translation.

Modules in this file include:
- NoamLR: A custom learning rate scheduler based on the Noam scheme as described in the 'Attention is All You Need' paper.
- save_model: Function for saving model checkpoints.
- load_checkpoint: Function for loading model checkpoints.
- save_stats_to_csv: Function for saving training statistics to a CSV file.
- plot_metrics: A function to plot training/validation loss and BLEU score over epochs.
- count_parameters: A function to count the number of trainable parameters in a PyTorch model.
- make_iwslt14_local_file: A function to download and save the IWSLT14 dataset in local files.

This module makes it easier to manage model training, handle checkpoints, visualize losses,
and preprocess datasets for machine translation tasks.
"""
import os
import json
import torch
import logging
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset


# ------------------ Learning Rate Schedulers ------------------ #

class NoamLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Implements the Noam learning rate schedule from 'Attention Is All You Need'.

    This scheduler increases the learning rate linearly for the first `warmup_steps`
    training steps, and then decreases it proportionally to the inverse square root
    of the step number.

    Learning rate at step t is computed as:
        lr = model_size^{-0.5} * min(t^{-0.5}, t * warmup_steps^{-1.5})

    Attributes:
        model_size (int): The dimensionality of the model (used for scaling the learning rate).
        warmup_steps (int): Number of steps to linearly increase the learning rate.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 model_size: int = 512,
                 warmup_steps: int = 4000,
                 last_epoch: int = -1):
        """Initializes the NoamLR scheduler.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            model_size (int, optional): Dimensionality of the model (default: 512).
            warmup_steps (int, optional): Number of warm-up steps (default: 4000).
            last_epoch (int, optional): The index of last epoch. Default: -1.
        """
        if model_size <= 0:
            raise ValueError("model_size must be a positive integer.")
        if warmup_steps <= 0:
            raise ValueError("warmup_steps must be a positive integer.")

        self.model_size   = model_size
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Computes the learning rate for the current step based on the Noam schedule.

        Returns:
            list: A list containing the learning rate for each parameter group.
        """
        step  = max(1, self._step_count)  # Avoid division by zero
        scale = self.model_size ** -0.5
        # Calculate the Noam learning rate based on the current step
        lr    = scale * min(step ** -0.5, step * (self.warmup_steps ** -1.5))
        return [lr for _ in self.base_lrs]


# --------------------- Logging --------------------- #

class LogLevel:
    """Defines standard logging levels using logging module's integer values."""
    DEBUG    = logging.DEBUG
    INFO     = logging.INFO
    WARNING  = logging.WARNING
    ERROR    = logging.ERROR
    CRITICAL = logging.CRITICAL


def set_logging_level(logging_level: int):
    """
        Configures the root logger with a specified integer level and simple format.

        Args:
            logging_level (int): The desired logging level as an integer
                                 (e.g., LogLevel.DEBUG, LogLevel.INFO).
                                 Defaults to WARNING if an unknown integer is provided.
    """
    # Remove previously configured logging handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    VALID_LOG_LEVELS = {
        logging.DEBUG,
        logging.INFO,
        logging.WARNING,
        logging.ERROR,
        logging.CRITICAL
    }

    # Validate log level, defaulting to WARNING if invalid.
    if not isinstance(logging_level, int) or logging_level not in VALID_LOG_LEVELS:
        valid_log_level = logging.WARNING
    else:
        valid_log_level = logging_level

    # Execute the basic configuration
    logging.basicConfig(level=valid_log_level, format='%(levelname)s - %(message)s')


# ------------------ Checkpointing ------------------ #

def save_model(
        epoch: int,
        model: torch.nn.Module,
        opt: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        loss: float,
        filepath: str ="model_checkpoint.pth"):
    """
    Save model checkpoint.

    Args:
        epoch (int): Current epoch number
        model (nn.Module): Model to save
        opt (torch.optim.Optimizer): Optimizer state to save
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler state to save
        loss (float): Current loss value
        filepath (str): Path to save the checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss
    }
    try:
        # Ensure the directory exists
        output_dir = os.path.dirname(filepath)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            logging.debug(f"Ensured directory exists: {output_dir}")

        torch.save(checkpoint, filepath)
        logging.info(f"Model checkpoint saved successfully at epoch {epoch} "
                     f"to {filepath}")
    except Exception as e:
        logging.error(f"Failed to save model checkpoint at epoch {epoch} "
                      f"to {filepath}: {e}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    checkpoint_path: str = "model_checkpoint.pth",
    device: torch.device = torch.device("cpu")) -> tuple[int, float | None]:
    """
    Loads model checkpoint, ignoring keys that are not present.
    """
    if not os.path.exists(checkpoint_path):
        logging.info(f"No checkpoint found at '{checkpoint_path}'. "
                     f"Starting training from epoch 1.")
        return 1, None

    try:
        logging.info(f"Attempting to load checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        logging.info("Model state loaded. Positional encoding buffer was skipped "
                     "as intended.")

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            logging.warning("Optimizer state not found in checkpoint.")

        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            logging.warning("Scheduler state not found in checkpoint.")

        start_epoch = checkpoint["epoch"] + 1
        last_loss = checkpoint.get("loss", None)

        logging.info(f"Successfully resumed model from epoch {start_epoch}. "
                     f"Last loss: {last_loss if last_loss is not None else 'N/A'}")

        return start_epoch, last_loss

    except Exception as e:
        logging.error(f"Failed to load checkpoint from '{checkpoint_path}': "
                      f"{e}. Starting from epoch 1.")
        return 1, None


def save_stats_to_csv(
        stats_record: dict[str, list[float]],
        file_path: str = None,
        epoch: int = None):
    """
    Saves training statistics to a CSV file.

    This function conditionally saves either the entire history of metrics
    or appends the latest epoch's metrics to an existing file.

    Args:
        stats_record (dict[str, list[float]]): A dictionary where each key is a
            metric name and each value is a list of metric values per epoch.
        file_path (str, optional): Path to the CSV file. If None, defaults to
            'training_stats.csv' in the current working directory.
        epoch (int, optional): If provided, the function appends only the latest
            values for the given epoch to the CSV. If None, it overwrites any
            existing file with the entire history from `stats_record`.

    Raises:
        ValueError: If `stats_record` contains no non-empty lists of metrics.
        OSError: If there's an issue creating the output directory or writing the
                 file.
    """
    target_path = file_path if file_path is not None else "training_stats.csv"

    available_data = {k: v for k, v in stats_record.items() if v}
    if not available_data:
        logging.warning("No non-empty stats data provided to save. Aborting save "
                        "operation.")
        raise ValueError("Cannot save stats: No non-empty data found in "
                         "'stats_record'.")

    output_dir = os.path.dirname(target_path)
    if output_dir:  # Checks if output_dir is not an empty string
        try:
            os.makedirs(output_dir, exist_ok=True)
            logging.debug(f"Ensured output directory exists: '{output_dir}'")
        except Exception as e:
            logging.error(f"Failed to create directory '{output_dir}': {e}")
            raise OSError(
                f"Could not create output directory '{output_dir}'.") from e

    try:
        if epoch is None:   # Full save mode
            num_epochs = len(next(iter(available_data.values())))

            # Create a DataFrame for all historical data
            df = pd.DataFrame(
                {'epoch': list(range(1, num_epochs + 1)), **available_data})
            df.to_csv(target_path, index=False)
            logging.info(f"Full training statistics (epochs 1-{num_epochs}) saved "
                         f"to: '{target_path}'")
        else:   # Append mode
            new_data = {
                'epoch': [epoch],
                **{k: [v[-1]] for k, v in available_data.items()}
            }
            # Create a DataFrame for the new data point
            df = pd.DataFrame(new_data)

            write_header = not os.path.exists(target_path)

            df.to_csv(target_path, mode='a', header=write_header, index=False)
            logging.info(f"Epoch {epoch} statistics appended to: '{target_path}'")

    except Exception as e:
        logging.error(f"Failed to save training statistics to '{target_path}': {e}")
        raise OSError(f"Could not write to file '{target_path}'.") from e


# ------------------ Visualization ------------------ #

def _plot_losses(statistics: dict[str, list[float]]):
    """
    Plots the training and validation loss on the same graph for direct comparison.

    Args:
        statistics (dict): A dictionary with two keys:
            - 'train' (list): Training loss values per epoch.
            - 'validation' (list): Validation loss values per epoch.

    The function creates a single plot:
    - The x-axis represents epochs.
    - The y-axis represents the loss values.
    - Both train and validation losses are plotted with different colors and markers.
    """
    if "train" not in statistics or "validation" not in statistics:
        logging.error("Input dictionary must contain 'train' and 'validation' keys "
                      "for _plot_losses.")
        raise ValueError("Input dictionary must contain 'train' and 'validation' "
                         "keys.")
    # --- Data Extraction ---
    train_loss = statistics['train']
    validation_loss = statistics['validation']
    epochs = range(1, len(train_loss) + 1)
    # --- Plotting Configuration ---
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, linestyle='-', color='#1f77b4',
             label='Train Loss', linewidth=2)
    plt.plot(epochs, validation_loss, linestyle='-', color='#d62728',
             label='Validation Loss', linewidth=2)
    # --- Chart Customization ---
    plt.title("Training & Validation Loss Over Epochs",
              fontsize=16, fontweight='bold')
    plt.xticks(epochs) # This ensures that xticks are integers
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    # --- Display Plot ---
    plt.show()


def _plot_bleu(bleu_scores: list[float]):
    """
    Plots BLEU scores over training epochs to visualize translation performance.

    Args:
        bleu_scores (list): A list of BLEU scores, one per epoch.

    The function creates a single plot:
    - The x-axis represents epochs.
    - The y-axis represents BLEU scores.
    - BLEU scores are plotted with a green line to show trends in translation quality.
    """
    if not bleu_scores:
        logging.warning("No BLEU scores provided to plot. Skipping BLEU plot.")
        return

    epochs = list(range(1, len(bleu_scores) + 1))
    # --- Plotting Configuration ---
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, bleu_scores, label='BLEU Score', color='green', linewidth=2)
    # --- Chart Customization ---
    plt.title("BLEU Score Over Epochs", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("BLEU Score", fontsize=12)
    plt.xticks(epochs)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    # --- Display Plot ---
    plt.show()


def plot_metrics(records: dict[str, list[float]]):
    """
    Plots training/validation losses and BLEU scores based on the provided metrics.

    Args:
        records (dict): A dictionary containing recorded metrics. Expected keys:
            - 'train' (list): Training loss values per epoch.
            - 'validation' (list): Validation loss values per epoch.
            - 'bleu' (list): BLEU scores per epoch.

    The function conditionally generates plots:
    - If both 'train' and 'validation' are present and non-empty, it plots losses.
    - If 'bleu' is present and non-empty, it plots BLEU scores.
    """
    # Plot training and validation losses if data is available
    if records.get('train') and records.get('validation'):
        _plot_losses(records)

    # Plot BLEU scores if data is available
    if records.get('bleu'):
        _plot_bleu(records['bleu'])


# ------------------ Data Preprocessing ------------------ #

def make_iwslt14_local_file(split: str,
                            debug: bool = False,
                            debug_size: int = 1000):
    """
    Saves the IWSLT14 dataset as a JSON file.

    Args:
        split (str): The dataset split to save ("train", "validation", or "test").
        debug (bool): If True, saves only a small subset for debugging.
        debug_size (int): Number of samples to keep in debug mode.
    """
    if split not in ["train", "validation", "test"]:
        logging.error(f"Invalid dataset split provided: '{split}'. Must be 'train', "
                      f"'validation', or 'test'.")
        raise ValueError(f"Invalid 'split' argument: '{split}'.")

    dataset = load_dataset("ahazeemi/iwslt14-en-fr")[split]

    if debug: # Debug mode
        if debug_size <= 0:
            logging.warning(
                f"Debug size '{debug_size}' is invalid. Using default of 100.")
            debug_size = 100  # Fallback for invalid debug_size
        logging.info(f"Debug mode enabled: Selecting {debug_size} samples from "
                     f"'{split}' split.")
        actual_debug_size = min(debug_size, len(dataset))
        dataset = dataset.select(range(actual_debug_size))
        logging.debug(f"Selected {actual_debug_size} samples for debug mode.")

    logging.debug(
        f"Dataset loaded. Total samples in '{split}' split: {len(dataset)}")

    # Save dataset under the correct split
    local_dataset = {
        split: {
            "en": dataset["en"],
            "fr": dataset["fr"]
        }
    }

    filename = f"iwslt14_{split}_debug.json" if debug else f"iwslt14_{split}.json"

    output_dir = "data/local_datasets"
    os.makedirs(output_dir, exist_ok=True)

    full_filepath = os.path.join(output_dir, filename)

    with open(full_filepath, "w", encoding="utf-8") as f:
        json.dump(local_dataset, f, ensure_ascii=False, indent=4)

    print(f"{split} dataset saved as {filename} ({'debug' if debug else 'full'})")


"""
In order to generate full and debug datasets for train, validation,
and test splits of IWSLT14 Fr-En, uncomment the code below and run it
"""
# for sp in ["train", "validation", "test"]:
#     make_iwslt14_local_file(split=sp, debug=False)  # Full dataset
#     make_iwslt14_local_file(split=sp, debug=True)  # Debug datasetstill



# def make_iwslt14_local_file(split: str,
#                             debug: bool = False,
#                             debug_size: int = 1000):
#     """
#     Saves the IWSLT14 dataset as a JSON file.
#
#     Args:
#         split (str): The dataset split to save ("train", "validation", or "test").
#         debug (bool): If True, saves only a small subset for debugging.
#         debug_size (int): Number of samples to keep in debug mode.
#
#     Raises:
#         ValueError: If an invalid dataset split is provided.
#         FileNotFoundError: If the output directory cannot be created.
#         IOError: If there's an issue writing the JSON file.
#     """
#     if split not in ["train", "validation", "test"]:
#         logging.error(
#             f"Invalid dataset split provided: '{split}'. Must be 'train', 'validation', or 'test'.")
#         raise ValueError(f"Invalid 'split' argument: '{split}'.")
#
#     try:
#         logging.info(
#             f"Loading IWSLT14 '{split}' split from 'ahazeemi/iwslt14-en-fr' dataset.")
#         dataset = load_dataset("ahazeemi/iwslt14-en-fr")[split]
#
#         if debug:
#             if debug_size <= 0:
#                 logging.warning(
#                     f"Debug size '{debug_size}' is invalid. Using default of 100.")
#                 debug_size = 100  # Fallback for invalid debug_size
#
#             logging.info(
#                 f"Debug mode enabled: Selecting {debug_size} samples from '{split}' split.")
#             # Ensure debug_size doesn't exceed dataset size
#             actual_debug_size = min(debug_size, len(dataset))
#             dataset = dataset.select(range(actual_debug_size))
#             logging.debug(f"Selected {actual_debug_size} samples for debug mode.")
#
#         logging.debug(
#             f"Dataset loaded. Total samples in '{split}' split: {len(dataset)}")
#
#     except KeyError:
#         logging.error(
#             f"'{split}' split not found in 'ahazeemi/iwslt14-en-fr' dataset. Please check the split name.")
#         raise
#     except Exception as e:
#         logging.error(f"Failed to load dataset for split '{split}': {e}")
#         raise
#
#     local_dataset = {
#         split: {
#             "en": dataset["en"],
#             "fr": dataset["fr"]
#         }
#     }
#
#     filename = f"iwslt14_{split}_debug.json" if debug else f"iwslt14_{split}.json"
#
#     # Ensure the output directory exists
#     output_dir = "data/local_datasets"  # Standardized output directory
#     os.makedirs(output_dir, exist_ok=True)
#
#     full_filepath = os.path.join(output_dir, filename)
#
#     try:
#         logging.info(
#             f"Saving '{split}' dataset to '{full_filepath}' ({'debug' if debug else 'full'}).")
#         with open(full_filepath, "w", encoding="utf-8") as f:
#             json.dump(local_dataset, f, ensure_ascii=False, indent=4)
#         logging.info(f"'{split}' dataset successfully saved to '{full_filepath}'.")
#     except IOError as e:
#         logging.error(f"Failed to save dataset to '{full_filepath}': {e}")
#         raise FileNotFoundError(
#             f"Could not write to file '{full_filepath}'. Check permissions or disk space.") from e
#     except Exception as e:
#         logging.error(
#             f"An unexpected error occurred while saving '{full_filepath}': {e}")
#         raise