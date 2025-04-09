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
from datasets import load_dataset
import matplotlib.pyplot as plt


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

    def __init__(self, optimizer, model_size=256, warmup_steps=4000, last_epoch=-1):
        """
        Initializes the NoamLR scheduler.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            model_size (int, optional): Dimensionality of the model (default: 256).
            warmup_steps (int, optional): Number of warm-up steps (default: 4000).
            last_epoch (int, optional): The index of last epoch. Default: -1.
        """
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Computes the learning rate for the current step based on the Noam schedule.

        Returns:
            list: A list containing the learning rate for each parameter group.
        """
        step = max(1, self._step_count)  # Avoid division by zero
        scale = self.model_size ** -0.5
        lr = scale * min(step ** -0.5, step * (self.warmup_steps ** -1.5))
        return [lr for _ in self.base_lrs]


def save_model(epoch, model, opt, scheduler, loss, filepath="model_checkpoint.pth"):
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
    print(f"Model checkpoint saved at epoch {epoch}.")


def load_checkpoint(model, optimizer, scheduler,
                    checkpoint_path="model_checkpoint.pth", device="cpu") -> int:
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
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        last_loss = checkpoint["loss"]

        print(f"Resuming model from epoch {start_epoch}")
        print(f"The last epoch loss: {last_loss}")
        return start_epoch
    return 1  # Start from epoch 1 if no checkpoint exists


def shift_trg_right(batch: torch.Tensor, eos_token_idx:  int = 3,
                    pad_token_idx: int = 1) -> torch.Tensor:
    """
    Convert <eos> token to <pad> token for each sentence in a batch of tensors.
    Designed for right-shift the target sequence in training transformer

    Args:
        batch (torch.Tensor): A batch of sentences (shape: [batch_size, seq_len]).
        eos_token_idx (int): The index of the <eos> token.
        pad_token_idx (int): The index of the <pad> token.

    Returns:
        torch.Tensor: The batch with <eos> replaced by <pad> for each sentence.
    """
    batch[batch == eos_token_idx] = pad_token_idx
    return batch


def plot_losses(loss_record: dict):
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
    train_loss = loss_record['train']
    validation_loss = loss_record['validation']
    epochs = range(1, len(train_loss) + 1)  # Assuming loss is recorded per epoch

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, linestyle='-', color='#1f77b4', label='Train Loss', linewidth=2)
    plt.plot(epochs, validation_loss, linestyle='-', color='#d62728', label='Validation Loss', linewidth=2)

    plt.title("Training & Validation Loss Over Epochs", fontsize=16, fontweight='bold')
    plt.xticks(epochs) # This ensures that xticks are integers
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.show()


def count_parameters(model: torch.nn.Module) -> int:
    """Returns the number of trainable parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The model whose parameters are to be counted.

    Returns:
        int: The total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



# print(f"Number of trainable parameters: {count_parameters(st_model):,}")


def make_iwslt14_local_file(split: str, debug: bool = False, debug_size: int = 1000):
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
        dataset = dataset.select(range(debug_size))  # Select only 100 samples for debugging

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
#     make_iwslt14_local_file(split=sp, debug=True)  # Debug dataset



# class NoamLR(torch.optim.lr_scheduler._LRScheduler):
#     """
#     Implements the Noam learning rate scheduler from 'Attention is All You Need'.
#
#     The learning rate increases linearly during warm-up and then decays as the
#     inverse square root of the step count after warm-up.
#
#     Attributes:
#         optimizer (torch.optim.Optimizer): The optimizer to apply the scheduler to.
#         model_size (int): The model size, used as a scaling factor.
#         warmup_steps (int): Number of warm-up steps before decay starts.
#         step_num (int): Tracks the current step count.
#     """
#
#     def __init__(self, optimizer, model_size=256, warmup_steps=4000, last_epoch=-1):
#         """Initializes the NoamLR scheduler.
#
#         Args:
#             optimizer (torch.optim.Optimizer): The optimizer to apply the scheduler to.
#             model_size (int, optional): The size of the model, used for scaling the learning rate.
#             warmup_steps (int, optional): Number of warm-up steps before decay starts.
#             last_epoch (int, optional): The index of the last epoch (for resuming training). Defaults to -1.
#         """
#         self.model_size = model_size
#         self.warmup_steps = warmup_steps
#         self.step_num = last_epoch + 1  # Tracks total steps (not epochs)
#         super().__init__(optimizer, last_epoch)
#
#     def get_lr(self) -> list:
#         """Computes the learning rate using the Noam schedule formula.
#
#         The learning rate follows:
#         lr = (model_size ** -0.5) * min(step_num ** -0.5, step_num * warmup_steps ** -1.5)
#
#         Returns:
#             list: A list containing the computed learning rate for each parameter group.
#         """
#         step = max(1, self.step_num)  # Prevent division by zero
#         scale = self.model_size ** -0.5
#         lr = scale * min(step ** -0.5, step * (self.warmup_steps ** -1.5))
#         return [lr for _ in self.base_lrs]
#
#     def step(self, epoch=None):
#         """Updates the learning rate at each optimizer step.
#
#         This method should be called after `optimizer.step()` to adjust the learning rate.
#
#         Args:
#             epoch (int, optional): Unused, included for compatibility with PyTorch's API.
#         """
#         self.step_num += 1
#         lr = self._get_noam_lr()
#         for param_group, new_lr in zip(self.optimizer.param_groups, lr):
#             param_group["lr"] = new_lr
