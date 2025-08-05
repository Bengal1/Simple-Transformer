"""
This module provides functions for training and evaluating a neural machine
translation model, specifically a Transformer. It includes utilities for
performing a single training epoch, managing the training loop across multiple
epochs, and integrating with evaluation metrics like validation loss and BLEU score.

The `_train_epoch` function handles the forward and backward passes for one
training iteration, including loss computation, gradient clipping, and optimizer/
scheduler steps. The `train_model` orchestrates the entire training process,
iterating through epochs, calling `_train_epoch`, performing validation,
and saving the best performing model.

Key functionalities:
- Training loop management with epoch-wise statistics recording.
- Integration with PyTorch's DataLoader, Optimizer, and LR Scheduler.
- Support for gradient clipping to prevent exploding gradients.
- Logging of training and validation progress.
- Model checkpointing based on validation loss improvement.
"""
import torch
import logging
import evaluation
import utils


def _train_epoch(model: torch.nn.Module,
                train_loader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler._LRScheduler,
                criterion: torch.nn.modules.loss,
                device: torch.device,
                max_gradient_clip: float,
                target_vocabulary_size: int) -> float:
    """
    Performs one epoch of training on the given model.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer used for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        criterion (torch.nn.modules.loss): Loss function.
        device (torch.device): Device to run training on.
        max_gradient_clip (float): Maximum gradient norm for clipping.
        target_vocabulary_size (int): Size of the target vocabulary.

    Returns:
        float: Average training loss over the epoch.
    """
    model.train()
    total_train_loss = 0.0

    for src, trg in train_loader:
        src, trg = src.to(device), trg.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(src, trg[:, :-1])  # Teacher forcing

        # Flatten the output and target tensors for loss computation
        logits = output.view(-1, target_vocabulary_size)
        targets = trg[:, 1:].contiguous().view(-1)

        # Compute loss
        loss = criterion(logits, targets)
        total_train_loss += loss.item()

        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_clip)

        # Scheduler & Optimizer steps
        optimizer.step()
        scheduler.step()

    return total_train_loss / len(train_loader)


def train_model(model: torch.nn.Module,
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
    best_loss = float('inf')

    logging.info(f"Starting model training for {epochs} epochs on {device}.")

    for epoch in range(start_epoch, epochs + 1):
        logging.info(f"--- Epoch {epoch}/{epochs} ---")
        # Train
        train_loss = _train_epoch(model, train_loader, optimizer, scheduler, criterion, 
                                  device, max_gradient_clip, target_vocabulary_size)
        stats_record['train'].append(train_loss)

        # Validation
        val_loss = evaluation.evaluate_model(model, validation_loader, criterion,
                                             device)
        stats_record['validation'].append(val_loss)

        # Evaluate BLEU
        bleu_score = evaluation.evaluate_bleu(model, validation_loader,
                                              target_vocabulary, device,
                                              special_tokens)
        stats_record['bleu'].append(bleu_score)

        # Display epoch training and validation metrics.
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f} | "
              f"Validation Loss: {val_loss:.4f} | BLEU Score: {bleu_score:.4f}")

        # Log training stats to a CSV.
        save_stats_to_csv(stats_record, epoch=epoch)

        # Save the model state if validation loss improves.
        if val_loss < best_loss:
            best_loss = val_loss
            utils.save_model(epoch, model, optimizer, scheduler, val_loss)

    return stats_record
