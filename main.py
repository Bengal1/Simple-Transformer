# ----------------------------------------------------------------------
# Copyright (c) 2025, Bengal1
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ----------------------------------------------------------------------
"""
main.py

Entry point for training and evaluating a SimpleTransformer model on the IWSLT14 dataset.

This script performs the following steps:
- Configures hyperparameters, device, and logging for the training session.
- Loads the IWSLT14 dataset and creates data loaders for training, validation, and testing.
- Initializes the SimpleTransformer model, loss function, optimizer, and learning rate scheduler.
- Executes the training loop, optionally resuming from a checkpoint.
- Evaluates the final trained model on the test dataset.
- Plots the training history, including loss and BLEU scores, for visualization.
"""
__author__="Bengal1"

import torch
from torch.utils.data import DataLoader
import logging
import utils
import config
import evaluation
from train import train_model
from data.iwslt14 import IWSLT14Dataset
from models.SimpleTransformer import SimpleTransformer


# ----------------------- Hyperparameters & Config ----------------------- #
# These parameters define the model's architecture, training process,
# optimizer settings, and general application behavior.
# --- Model Architecture ---
EMBED_DIM       = 512       # Embedding dimension
NUM_HEADS       = 8         # Number of attention heads
NUM_LAYERS      = 6         # Number of Encoder/Decoder layers
D_K             = 64        # Dimension for K-space
D_V             = 64        # Dimension for V-space
# --- Training Process ---
BATCH_SIZE      = 32        # Batch size
EPOCHS          = 10        # Number of epochs
MAX_GRAD_CLIP   = 1.0       # Max norm gradient (for gradient clipping)
DROPOUT         = 0.1       # Dropout probability
LABEL_SMOOTHING = 0.1       # Label smoothing parameter
# --- Optimizer Settings (Adam) ---
LEARNING_RATE   = 1e-3      # Initial learning rate
BETAS           = (0.9, 0.98) # Adam Optimizer beta coefficients
EPSILON         = 1e-9      # Optimizer's epsilon for numerical stability
WARMUP          = 50        # Scheduler warmup period (number of steps)
WEIGHT_DECAY    = 1e-5      # Weight decay parameter (L2 regularization)
# --- Application-Specific Settings ---
DATA_DEBUG_MODE = True      # Debug mode flag (enables/disables debug features)
LOGGING_LEVEL   = utils.LogLevel.WARNING # Initial logging verbosity level

# File paths for debugging (small subset of the dataset).
DEBUG_DATA_PATHS = {
    "train":      "data/local_datasets/iwslt14_train_debug.json",
    "validation": "data/local_datasets/iwslt14_validation_debug.json",
    "test":       "data/local_datasets/iwslt14_test_debug.json"
}
# File paths for full dataset
DATA_PATHS = {
    "train":      "data/local_datasets/iwslt14_train.json",
    "validation": "data/local_datasets/iwslt14_validation.json",
    "test":       "data/local_datasets/iwslt14_test.json"
}

# ------------------------ Device & Logging Setup ------------------------ #
# Set computation device (GPU/CPU)
device = utils.get_device()
# Set the logging level.
utils.set_logging_level(LOGGING_LEVEL)

# ----------------------------- Data Loading ----------------------------- #
# Initialize the train, validation and test dataset to data loader

# Selects debug/full datasets based on the flag.
if DATA_DEBUG_MODE:
    data_paths_to_use = DEBUG_DATA_PATHS
    logging.info("Using DEBUG dataset paths.")
else:
    data_paths_to_use = DATA_PATHS
    logging.info("Using FULL dataset paths.")

# Load Datasets
iwslt14_data = IWSLT14Dataset(data_paths_to_use)
train_dataset, val_dataset, test_dataset = iwslt14_data.get_datasets()

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Extract datasets parameters
src_vocab_size, trg_vocab_size = iwslt14_data.get_vocabularies_sizes()
src_vocab, trg_vocab           = iwslt14_data.get_vocabularies()
special_tokens                 = iwslt14_data.get_special_tokens_list()

# ----------------------------- Model Setup ------------------------------ #
# Initialize the Transformer model, loss function, optimizer and
# custom learning rate scheduler.

# Initialize the SimpleTransformer model.
model = SimpleTransformer(src_vocab_size, trg_vocab_size,
                          embed_dim=EMBED_DIM,
                          num_heads=NUM_HEADS,
                          num_layers=NUM_LAYERS,
                          d_k=D_K,
                          d_v=D_V,
                          dropout=DROPOUT).to(device)

# Initialize the CrossEntropyLoss criterion.
criterion = torch.nn.CrossEntropyLoss(ignore_index=iwslt14_data.get_padding_index(),
                                      label_smoothing=LABEL_SMOOTHING).to(device)

# Initialize the Adam optimizer.
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE,
                             betas=BETAS,
                             eps=EPSILON,
                             weight_decay=WEIGHT_DECAY)

# Initialize the NoamLR learning rate scheduler.
scheduler = utils.NoamLR(optimizer, model_size=EMBED_DIM, warmup_steps=WARMUP)


# ------------------------------ Main Entry ------------------------------ #
if __name__ == "__main__":
    # Set starting epoch for a new training session.
    start_epoch = 1

    # Optional: Load a pre-trained model or resume training from a checkpoint.
    # start_epoch, _ = utils.load_checkpoint(model=model, optimizer=optimizer,
    #                                        scheduler=scheduler, device=device)

    # Train the model and collect loss history
    eval_records = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        target_vocabulary=trg_vocab,
        special_tokens=special_tokens,
        target_vocabulary_size=trg_vocab_size,
        device=device,
        epochs=EPOCHS,
        max_gradient_clip=MAX_GRAD_CLIP,
        start_epoch=start_epoch,
    )

    # Evaluate model on the test dataset
    test_loss = evaluation.evaluate_model(model, test_loader, criterion, device)
    print(f"\nTest loss: {test_loss:.2f}\n")

    # Compute BLEU score
    bleu_score = evaluation.evaluate_bleu(model, test_loader, trg_vocab, device,
                                          iwslt14_data.get_special_tokens_list(),
                                          verbose=True)

    # Plot training and validation losses
    utils.plot_metrics(eval_records)

    # Optional: count model parameters
    # model.count_parameters()