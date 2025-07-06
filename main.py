"""
main.py

Entry point for training and evaluating the SimpleTransformer model on the IWSLT14 dataset.

This script performs the following steps:
- Loads the debug split of the IWSLT14 dataset
- Initializes the Transformer model and its components
- Trains the model and tracks training/validation losses
- Saves the best checkpoint during training
- Loads the best model checkpoint
- Evaluates the final model on the test split (loss and BLEU score)
- Plots training and validation loss curves
"""

import torch
from torch.utils.data import DataLoader
from models.SimpeTransformer import SimpleTransformer
from data.iwslt14 import IWSLT14Dataset
import evaluation
import utils
from train import train_model


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
weight_decay = 1e-4     # Weight decay (Lambda)
betas = (0.9, 0.98)     # Adam Optimizer betas
epsilon = 1e-9          # Optimizer epsilon
warmup = 3              # Scheduler warmup period
dropout = 0.1           # Dropout probability

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using', device, '\n')

# ----------------------- Data Loading ----------------------- #
# train_dataset = IWSLT14Dataset(split="train")
# val_dataset = IWSLT14Dataset(split="validation")
# test_dataset = IWSLT14Dataset(split="test")

# Local files
# train_dataset = IWSLT14Dataset(split="train",local_file="data/local_datasets/iwslt14_train.json")
# val_dataset = IWSLT14Dataset(split="validation",local_file="data/local_datasets/iwslt14_validation.json")
# test_dataset = IWSLT14Dataset(split="test",local_file="data/local_datasets/iwslt14_test.json")

# Debug #
train_dataset = IWSLT14Dataset(split="train",local_file="data/local_datasets/iwslt14_train_debug.json")
val_dataset = IWSLT14Dataset(split="validation",local_file="data/local_datasets/iwslt14_validation_debug.json")
test_dataset = IWSLT14Dataset(split="test",local_file="data/local_datasets/iwslt14_test_debug.json")
print()

# DataLoaders for batching
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ----------------------- Model Setup ----------------------- #
"""
Initialize the Transformer model, loss function, optimizer, and custom learning rate scheduler.
"""
src_vocab_size, trg_vocab_size = train_dataset.get_vocab_sizes()
model = SimpleTransformer(src_vocab_size, trg_vocab_size, embed_dim,
                             num_heads=num_heads, num_layers=num_layer,
                             d_k=d_k, d_v=d_v, dropout=dropout).to(device)

criterion = torch.nn.CrossEntropyLoss(
    ignore_index=train_dataset.get_padding_index(), label_smoothing=0.1
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             betas=betas, eps=epsilon, weight_decay=weight_decay)

scheduler = utils.NoamLR(optimizer, model_size=embed_dim, warmup_steps=warmup)


# ----------------------- Main Entry ----------------------- #
if __name__ == "__main__":
    # Train the model and collect loss history
    loss_records = train_model(model, train_loader, val_loader, optimizer, scheduler,
                criterion, device, epochs, max_grad_clip, trg_vocab_size)

    # Load the best checkpoint from training
    utils.load_checkpoint(st_model, optimizer, scheduler)

    # Evaluate model on the test dataset
    test_loss = evaluation.evaluate_model(st_model, test_loader, criterion, device)
    print(f"\nTest loss: {test_loss:.2f}\n")

    # Compute BLEU score
    bleu_score = evaluation.evaluate_bleu(
        st_model, test_loader, test_dataset.fr_vocab,
        device, train_dataset.get_special_tokens(), verbose=True
    )
    # Plot training and validation losses
    utils.plot_losses()

    # Optional: count model parameters
    # print(f"Number of trainable parameters: {utils.count_parameters(model):,}")