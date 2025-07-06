import torch
import evaluation
import utils


def train_epoch(model, train_loader, optimizer, scheduler, criterion,
                device, grad_clip, trg_vocab_size) -> float:
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

        optimizer.zero_grad()
        output = model(src, trg[:, :-1])  # Teacher forcing

        logits = output.view(-1, trg_vocab_size)
        targets = trg[:, 1:].contiguous().view(-1)

        loss = criterion(logits, targets)
        total_train_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

    return total_train_loss / len(train_loader)


def train_model(model, train_loader, val_loader, optimizer, scheduler,
                criterion, device, epochs, max_grad_clip, trg_vocab_size) -> dict[str, list[float]]:
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
    loss_record = {'train': [], 'validation': []}
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



# # Training loop
# def train_epoch(model, train_loader, optimizer, scheduler, criterion, device) -> float:
#     """
#     Performs one epoch of training on the model.
#
#     Returns:
#         float: The average loss for the epoch.
#     """
#     model.train()
#     total_train_loss = 0
#
#     for batch_idx, (src, trg) in enumerate(train_loader):
#         # Move data to device (GPU/CPU)
#         src, trg = src.to(device), trg.to(device)
#
#         # Forward pass
#         optimizer.zero_grad()
#         output = model(src, trg[:, :-1])  # Teacher forcing
#
#         # Flatten the output and target tensors for loss computation
#         logits = output.view(-1, trg_vocab_size)
#         targets = trg[:, 1:].contiguous().view(-1)
#
#         # Compute loss
#         loss = criterion(logits, targets)
#         total_train_loss += loss.item()
#
#         # Backpropagation and optimization
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_clip)
#         optimizer.step()
#
#         # Scheduler step - Update learning rate
#         scheduler.step()
#
#     avg_loss = total_train_loss / len(train_loader)
#
#     return avg_loss
#
#
# # Train #
# def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, device, epochs) -> dict:
#     """
#     Trains the model and evaluates it on the validation set after each epoch.
#
#     Returns:
#         dict: A dictionary containing the recorded losses for training and validation, with the keys:
#             - 'train' (list of float): Average training loss per epoch.
#             - 'validation' (list of float): Validation loss per epoch.
#     """
#     loss_record = {'train': [], 'validation': []}  # , 'bleu': []}
#
#     model.train()
#     best_loss = float('inf')
#
#     for epoch in range(1, epochs + 1):
#         train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
#         loss_record['train'].append(train_loss)
#
#         # Evaluate BLEU on the validation set
#         val_loss = evaluation.evaluate_model(model, val_loader, criterion, device)
#         loss_record['validation'].append(val_loss)
#         print(f"Epoch {epoch}: Train Loss: {train_loss:.4f} | "
#               f"Validation Loss: {val_loss:.4f}")
#
#         # Save the model if the loss is the best so far
#         if val_loss < best_loss:
#             best_loss = val_loss
#             utils.save_model(epoch, model, optimizer, scheduler, val_loss)
#
#     return loss_record
