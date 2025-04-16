import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.SimpeTransformer import SimpleTransformer
from data.iwslt14 import IWSLT14Dataset
import evaluation
import utils


# Hyperparameters
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
warmup = 5              # Scheduler warmup period
dropout = 0.1

# Set device #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using', device, '\n')

# Data #
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

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model #
src_vocab_size, trg_vocab_size = train_dataset.get_vocab_sizes()
max_length = train_dataset.get_max_length()
st_model = SimpleTransformer(src_vocab_size, trg_vocab_size, embed_dim,
                             num_heads=num_heads, num_layers=num_layer, d_k=d_k, d_v=d_v,
                             dropout=dropout).to(device)

# Loss & Optimization #
criterion = torch.nn.CrossEntropyLoss(ignore_index=train_dataset.get_padding_index(),
                                      label_smoothing=0.1).to(device)
optimizer = optim.Adam(st_model.parameters(), lr=learning_rate, betas=betas, eps=epsilon,
                        weight_decay=weight_decay)

# NoamLR Scheduler - Custom Scheduler
scheduler = utils.NoamLR(optimizer, model_size=embed_dim, warmup_steps=warmup)


# Training loop
def train() -> float:
    """
    Performs one epoch of training on the model.

    Returns:
        float: The average loss for the epoch.
    """
    st_model.train()
    total_train_loss = 0

    for batch_idx, (src, trg) in enumerate(train_loader):
        # Move data to device (GPU/CPU)
        src, trg = src.to(device), trg.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = st_model(src, trg[:, :-1])  # Teacher forcing

        # Flatten the output and target tensors for loss computation
        output_flat = output.view(-1, trg_vocab_size)
        trg_flat = trg[:, 1:].contiguous().view(-1)

        # Compute loss
        loss = criterion(output_flat, trg_flat)
        total_train_loss += loss.item()

        # Backpropagation and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(st_model.parameters(), max_grad_clip)
        optimizer.step()

        # Scheduler step - Update learning rate
        scheduler.step()

    avg_loss = total_train_loss / len(train_loader)

    return avg_loss


# Train #
def train_model() -> dict:
    """
    Trains the model and evaluates it on the validation set after each epoch.

    Returns:
        dict: A dictionary containing the recorded losses for training and validation, with the keys:
            - 'train' (list of float): Average training loss per epoch.
            - 'validation' (list of float): Validation loss per epoch.
    """
    loss_record = {'train': [], 'validation': []}

    st_model.train()
    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        # print(f"\nEpoch {epoch}/{epochs}")
        train_loss = train()
        loss_record['train'].append(train_loss)

        # Evaluate BLEU on the validation set
        val_loss = evaluation.evaluate_model(st_model, val_loader, criterion, device)
        loss_record['validation'].append(val_loss)
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f} | "
              f"Validation Loss: {val_loss:.4f}")

        # Save the model if the loss is the best so far
        if train_loss < best_loss:
            best_loss = train_loss
            utils.save_model(epoch, st_model, optimizer, scheduler, train_loss)

    return loss_record


# Entry point
if __name__ == "__main__":
    loss_records = train_model()

    # Load Best Checkpoint
    utils.load_checkpoint(st_model, optimizer, scheduler)

    # Test Model
    test_loss = evaluation.evaluate_model(st_model, test_loader, criterion,  device)
    print(f"\nTest loss: {test_loss:.2f}\n")

    # Test BLEU
    bleu_score = evaluation.evaluate_bleu(st_model, test_loader, test_dataset.fr_vocab,  device, verbose=True)
    # print(f"\nBLEU on test set: {bleu_score:.2f}")

    # Plot Train & Validation Loss
    utils.plot_losses(loss_records)


# print(f"Number of trainable parameters: {utils.count_parameters(st_model):,}")
