# ----------------------------------------------------------------------
# Copyright (c) 2025, Bengal1
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ----------------------------------------------------------------------
"""
main.py

Entry point for training and evaluating the SimpleTransformer model on the IWSLT14 dataset.

This script performs the following steps:
1. Loads dataset and builds DataLoaders.
2. Initializes model, loss function, optimizer, and learning rate scheduler.
3. Trains the model with optional checkpoint resume.
4. Evaluates the trained model on test data using loss and BLEU score.
5. Optionally counts model parameters and exits.
6. Plots training/validation metrics after training.

Usage:
    - Run a full training session:        main(config, device)
    - Resume from last checkpoint:        main(config, device, start_new=False)
    - Just count model parameters:        main(config, device, count_param_only=True)
"""
__author__="Bengal1"

import torch
from torch.utils.data import DataLoader
import logging
from utils import *
from config import Config
from scripts.evaluation import *
from scripts.train import train_model
from data.iwslt14 import IWSLT14Dataset
from models.SimpleTransformer import SimpleTransformer


def setup_data_loaders(cfg:Config) -> tuple:
    """
    Sets up the IWSLT14 datasets and PyTorch DataLoaders.

    This function determines whether to use the debug or full dataset
    based on the configuration, loads the data, and initializes
    DataLoaders for training, validation, and testing.

    Args:
        cfg (Config): A configuration object with dataset paths and training settings.

    Returns:
        tuple: A tuple containing the initialized data loaders
               (train_loader, val_loader, test_loader) and the
               IWSLT14Dataset object.
    """
    # Get dataset paths (if 'use_debug=True', it will return debug dataset paths)
    paths = cfg.dataset_paths.get()

    # Load Datasets
    iwslt14_data = IWSLT14Dataset(paths)
    train_dataset, val_dataset, test_dataset = iwslt14_data.get_datasets()

    # DataLoaders
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.training.batch_size,
                              num_workers=cfg.runtime.num_workers,shuffle=True)
    val_loader   = DataLoader(val_dataset,
                              batch_size=cfg.training.batch_size,
                              num_workers=cfg.runtime.num_workers, shuffle=False)
    test_loader  = DataLoader(test_dataset,
                              batch_size=cfg.training.batch_size,
                              num_workers=cfg.runtime.num_workers, shuffle=False)

    return train_loader, val_loader, test_loader, iwslt14_data


def setup_model_and_training(
        cfg: Config,
        iwslt14_data: IWSLT14Dataset,
        device: torch.device) -> tuple:
    """
    Initializes the model, loss function, optimizer, and learning rate scheduler.

    This function uses the provided configuration and dataset information to build
    all the necessary components for the training pipeline, including the
    SimpleTransformer model, the CrossEntropyLoss criterion, the Adam optimizer,
    and a custom NoamLR scheduler.

    Args:
        cfg (Config): A configuration object containing model and training hyperparameters.
        iwslt14_data (IWSLT14Dataset): The dataset object, used to get vocabulary sizes
                                       and padding index.
        device (torch.device): The device (CPU or GPU) where the model will be placed.

    Returns:
        tuple: A tuple containing the initialized model, criterion, optimizer,
               and scheduler.
    """
    # Extract datasets parameters for model
    src_vocab_size, trg_vocab_size = iwslt14_data.get_vocabularies_sizes()

    # Initialize the SimpleTransformer model.
    model = SimpleTransformer(src_vocab_size, trg_vocab_size,
                              embed_dim=cfg.model.embed_dim,
                              num_heads=cfg.model.num_heads,
                              num_layers=cfg.model.num_layers,
                              d_k=cfg.model.d_k,
                              d_v=cfg.model.d_v,
                              d_ff=cfg.model.d_ff,
                              dropout=cfg.model.dropout).to(device)

    # Initialize the CrossEntropyLoss loss_fn.
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=iwslt14_data.get_padding_index(),
                                        label_smoothing=cfg.training.label_smoothing
                                        ).to(device)

    # Initialize the Adam optimizer.
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg.training.learning_rate,
                                 betas=cfg.training.betas,
                                 eps=cfg.training.epsilon,
                                 weight_decay=cfg.training.weight_decay)

    # Initialize the NoamLR learning rate scheduler.
    scheduler = NoamLR(optimizer,
                       model_size=cfg.model.embed_dim,
                       warmup_steps=cfg.training.warmup_steps)

    return model, loss_fn, optimizer, scheduler


# --- Main ---
def main(
        cfg: Config,
        device: torch.device,
        start_new: bool = True,
        count_param_only: bool = False) -> None:
    """
    Main training and evaluation pipeline for the SimpleTransformer model.

    This function orchestrates the entire machine learning workflow:
    1. Sets up data loaders for the training, validation, and test datasets.
    2. Initializes the model, loss function, optimizer, and learning rate scheduler.
    3. Handles checkpoint loading to either resume training or start from scratch.
    4. Executes the training loop and saves evaluation records.
    5. Evaluates the trained model on the test set for both loss and BLEU score.
    6. Plots the training and validation metrics for visualization.

    Args:
        cfg (Config): A configuration object containing all hyperparameters and settings.
        device (torch.device): The device (CPU or GPU) on which to run the model.
        start_new (bool, optional): If True, training starts from the first epoch.
                                    If False, the script attempts to load the latest checkpoint
                                    to resume training. Defaults to True.
        count_param_only (bool, optional): If True, the function counts and prints the
                                            model's parameters and then exits without training.
                                            Defaults to False.
    """
    # Setup data loaders
    train_loader, val_loader, test_loader, iwslt14_data = setup_data_loaders(cfg)

    # Optional: Set a custom checkpoint path if provided.
    user_provided_path = None
    if user_provided_path:
        cfg.checkpoint.set_custom_path(user_provided_path)

    # Setup model and training components
    model, loss_fn, optimizer, scheduler = setup_model_and_training(cfg,
                                                                    iwslt14_data,
                                                                    device)
    # Extract datasets parameters for training
    _, trg_vocab_size    = iwslt14_data.get_vocabularies_sizes()
    src_vocab, trg_vocab = iwslt14_data.get_vocabularies()
    special_tokens       = iwslt14_data.get_special_tokens_list()

    if count_param_only:
        model.count_parameters()
        return

    # Set starting epoch
    start_epoch = 1 # Default
    if not start_new:
        try:
            # Load a pre-trained model or resume training from a checkpoint.
            start_epoch, _ = load_checkpoint(model, optimizer, scheduler,
                                             cfg.checkpoint.model_path,
                                             device)
        except FileNotFoundError:
            logging.warning("Checkpoint not found. Starting from scratch.")

    # Train the model and collect loss history
    logging.info("Starting training...")
    eval_records = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=loss_fn,
        target_vocabulary=trg_vocab,
        special_tokens=special_tokens,
        target_vocabulary_size=trg_vocab_size,
        device=device,
        epochs=cfg.training.epochs,
        patience=cfg.training.patience,
        accumulation_steps=cfg.training.accumulation_steps,
        max_gradient_clip=cfg.training.max_grad_clip,
        start_epoch=start_epoch,
    )

    # Load best model
    load_checkpoint(model, optimizer, scheduler,
                    cfg.checkpoint.model_path, device)

    # Evaluate model on the test dataset
    test_loss = evaluate_model(model, test_loader, loss_fn, device)

    # Compute BLEU score
    bleu_score = evaluate_bleu(model, test_loader, trg_vocab, device,
                               iwslt14_data.get_special_tokens_list(),
                               verbose=True)

    # Output final test loss and BLEU score
    print(f"\nTest loss: {test_loss:.3f} | BLEU Score: {bleu_score:.3f}\n")

    # Plot training and validation losses
    plot_metrics(eval_records)


# --- Entry Point ---
if __name__ == "__main__":
    # --- Load configuration ---
    config = Config()
    # --- Set seed (for reproducibility) ---
    set_seed(config.runtime.seed)
    # --- Configure logging ---
    config.runtime.set_logging_level(config.runtime.logging_level)

    # --- Set computation device (GPU/CPU) ---
    comp_device = get_device()

    # --- Run main function ---
    # To start a new training session:
    main(config, comp_device)

    # Resume training from the latest checkpoint:
    # main(config, comp_device, start_new=False)

    # Count model parameters and exit:
    # main(config, comp_device, count_param_only=True)