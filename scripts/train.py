"""
This module contains the essential logic for the model's training pipeline.
It defines the complete multi-epoch training and validation process,
handling the core training loop, performance metrics, and checkpointing.

The module provides two main functions:
- `_train_epoch`: A private helper function that executes a single training
  epoch, managing the forward pass, backpropagation, and optimizer steps.
- `train_model`: The main public function that orchestrates the entire training
  workflow over multiple epochs. It calls `_train_epoch`, performs validation,
  calculates metrics (including BLEU score), and saves the model's state.
"""
import torch
import logging
from utils import save_model, save_stats_to_csv, early_stopping
from scripts.evaluation import evaluate_model, evaluate_bleu


# --- Public API ---
__all__ = ["train_model"]


def train_model(
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        validation_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        criterion: torch.nn.modules.loss,
        target_vocabulary: dict,
        special_tokens: list,
        target_vocabulary_size: int,
        device: torch.device,
        epochs: int = 10,
        patience: int = 5,
        accumulation_steps: int = 1,
        max_gradient_clip: float = 1.0,
        start_epoch: int = 1) -> dict[str, list[float]]:
    """
    Trains the model for multiple epochs and evaluates it on the validation set.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        validation_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): Optimizer used for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        criterion (torch.nn.modules.loss): Loss function.
        target_vocabulary (dict): Target language vocabulary
        special_tokens (list): List of the special tokens
        target_vocabulary_size (int): Size of the target vocabulary.
        device (torch.device): Device to run training on.
        epochs (int): Number of training epochs.
        patience (int, optional): Number of epochs to wait for improvement. Default is 5.
        accumulation_steps(int, optional): Steps to accumulate gradients before
                                        optimizer update. Default is 1.
        max_gradient_clip (float): Maximum gradient norm for clipping.
        start_epoch (int): The starting epoch for training. (Default is 1).

    Returns:
        dict[str, list[float]]: Dictionary with epoch-level training and validation
                                losses.
            Keys:
                - 'train': List of training loss values.
                - 'validation': List of validation loss values.
                - 'bleu': List of bleu score values.
    """
    stats_record = {'train': [], 'validation': [], 'bleu': []}
    best_bleu = float('-inf')

    logging.info(f"Starting model training for {epochs} epochs on {device}.")

    for epoch in range(start_epoch, epochs + 1):
        logging.info(f"--- Epoch {epoch}/{epochs} ---")
        # Train
        train_loss = _train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device,
            max_gradient_clip, target_vocabulary_size, accumulation_steps
        )
        stats_record['train'].append(train_loss)

        # Validation
        val_loss = evaluate_model(model, validation_loader, criterion, device)
        stats_record['validation'].append(val_loss)

        # Evaluate BLEU
        bleu_score = evaluate_bleu(model, validation_loader, target_vocabulary,
                                   device, special_tokens)
        stats_record['bleu'].append(bleu_score)

        # Display epoch training and validation metrics.
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f} | "
              f"Validation Loss: {val_loss:.4f} | BLEU Score: {bleu_score:.4f}")

        # Log training stats to a CSV.
        save_stats_to_csv(stats_record, epoch=epoch)

        # Save the model state if validation loss improves.
        if bleu_score > best_bleu:
            best_bleu = bleu_score
            save_model(epoch, model, optimizer, scheduler, val_loss)

        # Monitor BLEU plateau for potential early stopping
        if early_stopping(stats_record['bleu'], patience=patience):
            logging.info(f"Early stopping triggered at epoch {epoch}")
            break

    return stats_record


 # --- Training Helper Functions ---
def _train_epoch(
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        criterion: torch.nn.modules.loss,
        device: torch.device,
        max_gradient_clip: float,
        target_vocab_size: int,
        accumulation_steps: int = 1
) -> float:
    """
    Performs one epoch of training on the given model with gradient accumulation.

    Gradient accumulation allows simulating a larger batch size by accumulating
    gradients over multiple smaller batches before performing an optimizer step.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        optimizer (Optimizer): Optimizer used for updating model parameters.
        scheduler (_LRScheduler): Learning rate scheduler.
        criterion (nn.modules.loss): Loss function.
        device (torch.device): Device to run the training on.
        max_gradient_clip (float): Maximum gradient norm for clipping.
        target_vocab_size (int): Size of the target vocabulary (for reshaping logits).
        accumulation_steps(int, optional): Steps to accumulate gradients before
                                            optimizer update. Default is 1.

    Returns:
        float: Average training loss over the epoch (per batch, not per accumulated step).
    """
    model.train()
    running_loss = 0.0  # Sum of batch losses for reporting

    optimizer.zero_grad()  # Reset gradients at the start of the epoch

    batch_idx = -1
    for batch_idx, (src_batch, trg_batch) in enumerate(train_loader):
        src_batch, trg_batch = src_batch.to(device), trg_batch.to(device)

        # Forward pass
        predictions = model(src_batch, trg_batch[:, :-1])  # Teacher forcing

        # Flatten predictions and targets for loss computation
        logits = predictions.view(-1, target_vocab_size)
        targets = trg_batch[:, 1:].contiguous().view(-1)

        # Compute loss for current batch and scale by accumulation_steps
        loss = criterion(logits, targets) / accumulation_steps
        running_loss += (
                    loss.item() * accumulation_steps)  # accumulate unscaled loss for reporting

        # Backpropagate scaled loss
        loss.backward()

        # Perform optimizer step every `accumulation_steps` batches
        if (batch_idx + 1) % accumulation_steps == 0:
            # Clip gradients to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_clip)

            # Update parameters and learning rate
            optimizer.step()
            scheduler.step()

            # Reset gradients after update
            optimizer.zero_grad()

    # Handle remaining gradients if number of batches is not divisible by accumulation_steps
    if (batch_idx + 1) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # Return average batch loss over the epoch
    return running_loss / len(train_loader)


