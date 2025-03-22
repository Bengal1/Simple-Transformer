import json
from datasets import load_dataset
import matplotlib.pyplot as plt


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
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.show()


# def plot_training_progress(history):
#     epochs_range = range(1, len(history["train_loss"]) + 1)
#
#     plt.figure(figsize=(12, 5))
#
#     # Plot Loss
#     plt.subplot(1, 2, 1)
#     plt.plot(epochs_range, history["train_loss"], label="Train Loss", marker="o", color="blue")
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss")
#     plt.title("Training Loss per Epoch")
#     plt.legend()
#
#     # Plot BLEU and ROUGE Scores
#     plt.subplot(1, 2, 2)
#     plt.plot(epochs_range, history["bleu"], label="BLEU", marker="o", color="green", linestyle='--')
#     plt.plot(epochs_range, history["rouge1"], label="ROUGE-1", marker="o", color="red", linestyle='-.')
#     plt.plot(epochs_range, history["rougeL"], label="ROUGE-L", marker="o", color="purple", linestyle=':')
#
#     plt.xlabel("Epochs")
#     plt.ylabel("Score")
#     plt.title("BLEU and ROUGE Scores per Epoch")
#     plt.legend()
#
#     plt.tight_layout()
#     plt.grid(True)
#     plt.show()


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
