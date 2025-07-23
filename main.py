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
embed_dim = 512         # Embedding dimension
num_heads = 8           # Number of attention heads
num_layer = 6           # Number of Encoder/Decoder layers
d_k = 64                # Dimension for K-space
d_v = 64                # Dimension for V-space
batch_size = 32         # Batch size
epochs = 10             # Number of epochs
max_grad_clip = 1.0     # Max norm gradient
learning_rate = 1e-3    # Learning rate
betas = (0.9, 0.98)     # Adam Optimizer betas
epsilon = 1e-9          # Optimizer epsilon
warmup = 20             # Scheduler warmup period
dropout = 0.1           # Dropout probability
label_smoothing = 0.1   # Label smoothing parameter
APP_DEBUG_MODE = False  # Debug mode variable (Flag)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using', device, '\n')

if APP_DEBUG_MODE:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
else:
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Application started. Logging configured globally.")

# ----------------------- Data Loading ----------------------- #
"""
Initialize the  train, validation and test dataset to data loader
"""
debug_datapath = {"train": "data/local_datasets/iwslt14_train_debug.json",
                  "validation": "data/local_datasets/iwslt14_validation_debug.json",
                  "test": "data/local_datasets/iwslt14_test_debug.json"}
datapath = {"train": "data/local_datasets/iwslt14_train.json",
            "validation": "data/local_datasets/iwslt14_validation.json",
            "test": "data/local_datasets/iwslt14_test.json"}

iwslt14_data = IWSLT14Dataset(debug_datapath)
train_dataset, val_dataset, test_dataset = iwslt14_data.get_datasets()

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Dataset parameters
src_vocab_size, trg_vocab_size = iwslt14_data.get_vocabularies_sizes()
src_vocab, trg_vocab = iwslt14_data.get_vocabularies()
special_tokens = iwslt14_data.get_special_tokens_list()
# ----------------------- Model Setup ----------------------- #
"""
Initialize the Transformer model, loss function, optimizer 
and custom learning rate scheduler.
"""

model = SimpleTransformer(src_vocab_size, trg_vocab_size, embed_dim,
                          num_heads=num_heads, num_layers=num_layer,
                          d_k=d_k, d_v=d_v, dropout=dropout).to(device)

criterion = torch.nn.CrossEntropyLoss(
    ignore_index=iwslt14_data.get_padding_index(),
    label_smoothing=label_smoothing).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             betas=betas, eps=epsilon)

scheduler = utils.NoamLR(optimizer, model_size=embed_dim, warmup_steps=warmup)

# ----------------------- Main Entry ----------------------- #
if __name__ == "__main__":
    # Train the model and collect loss history
    eval_records = train_model(model, train_loader, val_loader, optimizer,
                               scheduler, criterion, trg_vocab, special_tokens,
                               trg_vocab_size, device, epochs, max_grad_clip)

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
    # print(f"Number of trainable parameters: {utils.count_parameters(model):,}")