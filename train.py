import torch
import evaluation
import utils


def train_epoch(model: torch.nn.Module,
                train_loader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler._LRScheduler,
                criterion: torch.nn.modules.loss,
                device: torch.device,
                grad_clip: float,
                trg_vocab_size: int) -> float:
    """
    Performs one epoch of training on the given model.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer used for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to run training on.
        grad_clip (float): Maximum gradient norm for clipping.
        trg_vocab_size (int): Size of the target vocabulary.

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
        logits = output.view(-1, trg_vocab_size)
        targets = trg[:, 1:].contiguous().view(-1)

        # Compute loss
        loss = criterion(logits, targets)
        total_train_loss += loss.item()

        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # Scheduler & Optimizer steps
        optimizer.step()
        scheduler.step()

    return total_train_loss / len(train_loader)


def train_model(model: torch.nn.Module,
                train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler._LRScheduler,
                criterion: torch.nn.modules.loss,
                device: torch.device,
                epochs: int,
                max_grad_clip: float,
                trg_vocab_size: int) -> dict[str, list[float]]:
    """
    Trains the model for multiple epochs and evaluates it on the validation set.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): Optimizer used for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to run training on.
        epochs (int): Number of training epochs.
        max_grad_clip (float): Maximum gradient norm for clipping.
        trg_vocab_size (int): Size of the target vocabulary.

    Returns:
        dict[str, list[float]]: Dictionary with epoch-level training and validation losses.
            Keys:
                - 'train': List of training loss values.
                - 'validation': List of validation loss values.
    """
    loss_record = {'train': [], 'validation': []}  # , 'bleu': []}

    model.train()
    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                 criterion, device, max_grad_clip, trg_vocab_size)
        loss_record['train'].append(train_loss)

        val_loss = evaluation.evaluate_model(model, val_loader, criterion, device)
        loss_record['validation'].append(val_loss)

        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            utils.save_model(epoch, model, optimizer, scheduler, val_loss)

    return loss_record
