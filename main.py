"""
main.py

Entry point for training and evaluating the SimpleTransformer model on the IWSLT14 dataset.

This script performs the following steps:
- Loads the debug split of the IWSLT14 dataset
- Initializes the Transformer model and its components
- Trains the model, tracks training/validation losses and BLEU score
- Saves the best checkpoint during training
- Loads the best model checkpoint
- Evaluates the final model on the test split (loss and BLEU score)
- Plots training and validation loss curves
- Plot BLEU score curve
"""

import torch
from torch.utils.data import DataLoader
from models.SimpleTransformer import SimpleTransformer
from data.iwslt14 import IWSLT14Dataset
import evaluation
import utils
from train import train_model
import logging


# ------------------ Hyperparameters & Config ------------------ #
EMBED_DIM = 512         # Embedding dimension
NUM_HEADS = 8           # Number of attention heads
NUM_LAYERS = 6           # Number of Encoder/Decoder layers
D_K = 64                # Dimension for K-space
D_V = 64                # Dimension for V-space
BATCH_SIZE = 32         # Batch size
EPOCHS = 15             # Number of epochs
MAX_GRAD_CLIP = 1.0     # Max norm gradient
LEARNING_RATE = 1e-3    # Learning rate
BETAS = (0.9, 0.98)     # Adam Optimizer betas
EPSILON = 1e-9          # Optimizer's epsilon
WARMUP = 50             # Scheduler warmup period
DROPOUT = 0.1           # Dropout probability
LABEL_SMOOTHING = 0.1   # Label smoothing parameter
APP_DEBUG_MODE = False  # Debug mode variable (Flag)

DEBUG_DATA_PATHS = {"train": "data/local_datasets/iwslt14_train_debug.json",
                  "validation": "data/local_datasets/iwslt14_validation_debug.json",
                  "test": "data/local_datasets/iwslt14_test_debug.json"}
DATA_PATHS = {"train": "data/local_datasets/iwslt14_train.json",
            "validation": "data/local_datasets/iwslt14_validation.json",
            "test": "data/local_datasets/iwslt14_test.json"}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using', device, '\n') # DEBUG without logging - Remove!

if APP_DEBUG_MODE:
    log_level = logging.DEBUG
else:
    log_level = logging.WARNING

logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Application started. Logging configured globally.")
# logging.info(f"Using device: {device}")

# ----------------------- Data Loading ----------------------- #
"""
Initialize the  train, validation and test dataset to data loader
"""

# if APP_DEBUG_MODE: # Or based on a dedicated 'data_mode' argument
#     data_paths_to_use = DEBUG_DATA_PATHS
#     logging.info("Using DEBUG dataset paths.")
# else:
#     data_paths_to_use = DATA_PATHS
#     logging.info("Using FULL dataset paths.")

data_paths_to_use = DEBUG_DATA_PATHS # DEBUG without logging - Remove!

iwslt14_data = IWSLT14Dataset(data_paths_to_use)
train_dataset, val_dataset, test_dataset = iwslt14_data.get_datasets()

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Datasets parameters
src_vocab_size, trg_vocab_size = iwslt14_data.get_vocabularies_sizes()
src_vocab, trg_vocab = iwslt14_data.get_vocabularies()
special_tokens = iwslt14_data.get_special_tokens_list()

# ----------------------- Model Setup ----------------------- #
"""
Initialize the Transformer model, loss function, optimizer 
and custom learning rate scheduler.
"""

model = SimpleTransformer(src_vocab_size, trg_vocab_size, EMBED_DIM,
                          num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
                          d_k=D_K, d_v=D_V, dropout=DROPOUT).to(device)

criterion = torch.nn.CrossEntropyLoss(
    ignore_index=iwslt14_data.get_padding_index(),
    label_smoothing=LABEL_SMOOTHING).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                             betas=BETAS, eps=EPSILON)

scheduler = utils.NoamLR(optimizer, model_size=EMBED_DIM, warmup_steps=WARMUP)

# ----------------------- Main Entry ----------------------- #
if __name__ == "__main__":
    # Train the model and collect loss history
    eval_records = train_model(model, train_loader, val_loader, optimizer,
                               scheduler, criterion, trg_vocab, special_tokens,
                               trg_vocab_size, device, EPOCHS, MAX_GRAD_CLIP)

    # Load the best checkpoint from training
    # utils.load_checkpoint(model, optimizer, scheduler)

    # Evaluate model on the test dataset
    test_loss = evaluation.evaluate_model(model, test_loader, criterion, device)
    print(f"\nTest loss: {test_loss:.2f}\n")

    # Compute BLEU score
    bleu_score = evaluation.evaluate_bleu(
        model, test_loader, trg_vocab,
        device, iwslt14_data.get_special_tokens_list(), verbose=True)

    # Plot training and validation losses
    utils.plot_metrics(eval_records)

    # Optional: count model parameters
    # utils.count_parameters(model)