"""
This module provides utility functions and classes for machine learning tasks,
including model checkpoint saving/loading, learning rate scheduling, loss plotting,
and dataset preprocessing for machine translation.
    
Modules in this file include:
- NoamLR: A custom learning rate scheduler based on the Noam scheme as described in the 'Attention is All You Need' paper.
- save_model: Function for saving model checkpoints.
- load_checkpoint: Function for loading model checkpoints.
- shift_trg_right: A function to right-shift the target sequence during training in a transformer model.
- plot_losses: A function to plot training and validation loss over epochs.
- count_parameters: A function to count the number of trainable parameters in a PyTorch model.
- make_iwslt14_local_file: A function to download and save the IWSLT14 dataset in local files.
    
This module makes it easier to manage model training, handle checkpoints, visualize losses,
and preprocess datasets for machine translation tasks.
"""

import os
import json
import torch
import pandas as pd
from datasets import load_dataset
import matplotlib.pyplot as plt
from data.iwslt14 import IWSLT14Dataset
from torch.utils.data import DataLoader
from typing import Optional


class NoamLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Implements the Noam learning rate schedule from 'Attention Is All You Need'.

    This scheduler increases the learning rate linearly for the first `warmup_steps` training steps,
    and then decreases it proportionally to the inverse square root of the step number.

    Learning rate at step t is computed as:
        lr = model_size^{-0.5} * min(t^{-0.5}, t * warmup_steps^{-1.5})

    Attributes:
        model_size (int): The dimensionality of the model (used for scaling the learning rate).
        warmup_steps (int): Number of steps to linearly increase the learning rate.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 model_size: int = 256,
                 warmup_steps: int = 4000,
                 last_epoch: int = -1):
        """Initializes the NoamLR scheduler.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            model_size (int, optional): Dimensionality of the model (default: 256).
            warmup_steps (int, optional): Number of warm-up steps (default: 4000).
            last_epoch (int, optional): The index of last epoch. Default: -1.
        """
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list:
        """Computes the learning rate for the current step based on the Noam schedule.

        Returns:
            list: A list containing the learning rate for each parameter group.
        """
        step = max(1, self._step_count)  # Avoid division by zero
        scale = self.model_size ** -0.5
        lr = scale * min(step ** -0.5, step * (self.warmup_steps ** -1.5))
        return [lr for _ in self.base_lrs]


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
    torch.save(checkpoint, filepath)
    # print(f"Model checkpoint saved at epoch {epoch}.")


def load_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        checkpoint_path: str = "model_checkpoint.pth",
        device: torch.device = "cpu") -> int:
    """
    Load model checkpoint.

    Args:
        model (nn.Module): Model to load weights into
        optimizer (torch.optim.Optimizer): Optimizer to load state into
        scheduler (torch.optim.lr_scheduler): Scheduler to load state into
        checkpoint_path (str): Path to the checkpoint file
        device (str): Device to load model onto (default: "cpu")

    Returns:
        int: Start epoch number
    """
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        last_loss = checkpoint["loss"]

        # print(f"Resuming model from epoch {start_epoch}")
        # print(f"The last epoch loss: {last_loss}")
        return start_epoch
    return 1  # Start from epoch 1 if no checkpoint exists


def save_stats_to_csv(
    stats_record: dict[str, list[float]],
    file_path: str = None,
    epoch: int = None):
    """
    Saves training statistics to a CSV file.

    Args:
        stats_record (dict[str, list[float]]): A dictionary where each key is a metric name
            and each value is a list of metric values per epoch.
        file_path (str, optional): Path to the CSV file. Defaults to 'training_stats.csv'
            in the current directory.
        epoch (int, optional): If provided, appends only the latest values for the given
            epoch. If None, writes the entire history and overwrites any existing file.

    Raises:
        ValueError: If there is no non-empty data to save.
    """
    if file_path is None:
        file_path = "training_stats.csv"

    # Filter out empty metric lists
    available_data = {k: v for k, v in stats_record.items() if v}
    if not available_data:
        raise ValueError("No non-empty stats data to save.")

    if epoch is None:
        # Full save: assumes all lists are the same length
        num_epochs = len(next(iter(available_data.values())))
        df = pd.DataFrame({'epoch': list(range(num_epochs)), **available_data})
        df.to_csv(file_path, index=False)
    else:
        # Append mode: only the latest value for each stat
        new_data = {
            'epoch': [epoch],
            **{k: [v[-1]] for k, v in available_data.items()}
        }
        df = pd.DataFrame(new_data)
        df.to_csv(file_path, mode='a', header=not os.path.exists(file_path),
                  index=False)


def _plot_losses(statistics: dict[str, list[float]]):
    """
    Plots the training and validation loss on the same graph for direct comparison.

    Args:
        loss_record (dict): A dictionary with two keys:
            - 'train' (list): Training loss values per epoch.
            - 'validation' (list): Validation loss values per epoch.

    The function creates a single plot:
    - The x-axis represents epochs.
    - The y-axis represents the loss values.
    - Both train and validation losses are plotted with different colors and markers.
    """
    train_loss = statistics['train']
    validation_loss = statistics['validation']
    epochs = range(1, len(train_loss) + 1)  # Assuming loss is recorded per epoch

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, linestyle='-', color='#1f77b4',
             label='Train Loss', linewidth=2)
    plt.plot(epochs, validation_loss, linestyle='-', color='#d62728',
             label='Validation Loss', linewidth=2)

    plt.title("Training & Validation Loss Over Epochs",
              fontsize=16, fontweight='bold')
    plt.xticks(epochs) # This ensures that xticks are integers
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.show()  # <--- Bug!!!


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
    epochs = list(range(1, len(bleu_scores) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, bleu_scores, label='BLEU Score', color='green', linewidth=2)

    plt.title("BLEU Score Over Epochs", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("BLEU Score", fontsize=12)
    plt.xticks(epochs)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()

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
    if records.get('train') and records.get('validation'):
        _plot_losses(records)
    if records.get('bleu'):
        _plot_bleu(records['bleu'])


def count_parameters(model: torch.nn.Module) -> int:
    """
    Returns the number of trainable parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The model whose parameters are to be counted.

    Returns:
        int: The total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_iwslt14_local_file(split: str,
                            debug: bool = False,
                            debug_size: int = 1000):
    """
    Saves the IWSLT14 dataset as a JSON file.

    Args:
        split (str): The dataset split to save ("train", "validation", or "test").
        debug (bool): If True, saves only a small subset (e.g., 100 examples) for debugging.
        debug_size (int): Number of samples to keep in debug mode.
    """
    dataset = load_dataset("ahazeemi/iwslt14-en-fr")[split]
    # debug mode is enabled
    if debug:
        dataset = dataset.select(range(debug_size))

    # Save dataset under the correct split
    local_dataset = {
        split: {
            "en": dataset["en"],
            "fr": dataset["fr"]
        }
    }

    filename = f"iwslt14_{split}_debug.json" if debug else f"iwslt14_{split}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(local_dataset, f, ensure_ascii=False, indent=4)

    print(f"{split} dataset saved as {filename} ({'debug' if debug else 'full'})")


"""
In order to generate full and debug datasets for train, validation, 
and test splits of IWSLT14 Fr-En, uncomment the code below and run it
"""
# for sp in ["train", "validation", "test"]:
#     make_iwslt14_local_file(split=sp, debug=False)  # Full dataset
#     make_iwslt14_local_file(split=sp, debug=True)  # Debug datasetstill