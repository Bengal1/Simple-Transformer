import json
import torch
from datasets import load_dataset
import matplotlib.pyplot as plt
import os


def save_model(epoch, model, opt, loss, checkpoint_path="model_checkpoint.pth"):
    """
    Saves the model and optimizer states, along with the current epoch and loss,
    to a checkpoint file.

    Args:
        epoch (int): The current training epoch.
        model (torch.nn.Module): The model whose state will be saved.
        opt (torch.optim.Optimizer): The optimizer whose state will be saved.
        loss (float): The loss value at the time of saving.
        checkpoint_path (str, optional): The file path to save the checkpoint.
                                         Defaults to "model_checkpoint.pth".
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Model checkpoint saved at epoch {epoch}.")


def load_checkpoint(model, optimizer, checkpoint_path="model_checkpoint.pth"):
    """
    Loads a model checkpoint from the specified path and restores the model
    and optimizer states. If a checkpoint exists, training resumes from the
    saved epoch; otherwise, training starts from scratch.

    Args:
        model (torch.nn.Module): The model whose state will be loaded.
        optimizer (torch.optim.Optimizer): The optimizer whose state will be restored.
        checkpoint_path (str, Optional): Path to the checkpoint file.
                                         Defaults to "model_checkpoint.pth".

    Returns:
        int: The epoch number to resume training from (starting from the next epoch).
             Returns 0 if no checkpoint is found, indicating fresh training.
    """
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
        return start_epoch
    return 0  # Start fresh if no checkpoint


def shift_trg_right(batch, eos_token_idx=3, pad_token_idx=1):
    """
    Convert <eos> token to <pad> token for each sentence in a batch of tensors.
    Designed for right-shift the target sequence in training transformer

    Args:
        batch (Tensor): A batch of sentences (shape: [batch_size, seq_len]).
        eos_token_idx (int): The index of the <eos> token.
        pad_token_idx (int): The index of the <pad> token.

    Returns:
        Tensor: The batch with <eos> replaced by <pad> for each sentence.
    """
    # Replace eos_token_idx with pad_token_idx in the batch
    batch[batch == eos_token_idx] = pad_token_idx
    return batch


def plot_losses(loss_record):
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
    plt.plot(epochs, train_loss, marker='o', linestyle='-', color='#1f77b4', label='Train Loss', linewidth=2)
    plt.plot(epochs, validation_loss, marker='s', linestyle='-', color='#d62728', label='Validation Loss', linewidth=2)

    plt.title("Training & Validation Loss Over Epochs", fontsize=16, fontweight='bold')
    plt.xticks(epochs) # This ensures that xticks are integers
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.show()


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

# Generate full and debug datasets for train, validation, and test splits
# for sp in ["train", "validation", "test"]:
#     make_iwslt14_local_file(split=sp, debug=False)  # Full dataset
#     make_iwslt14_local_file(split=sp, debug=True)  # Debug dataset
