import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.SimpeTransformer import SimpleTransformer
from data.iwslt14 import IWSLT14Dataset
import evaluation
import utils
import config
# import time


torch.autograd.set_detect_anomaly(True)
# Hyperparameters
embed_dim = 256          # Embedding dimension
num_heads = 8            # Number of attention heads
d_k = 32                 # Dimension for K-space
d_v = 128                # Dimension for V-space (attention output)
batch_size = 32          # Batch size
epochs = 10              # Number of epochs
learning_rate = 1e-4     # Learning rate

# Set device #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using', device, '\n')

# Data #
# train_dataset = IWSLT14Dataset(split="train")
# val_dataset = IWSLT14Dataset(split="validation")
# test_dataset = IWSLT14Dataset(split="test")

# Local files
# train_dataset = IWSLT14Dataset(split="train",local_file="iwslt14_train.json")
# val_dataset = IWSLT14Dataset(split="validation",local_file="iwslt14_validation.json")
# test_dataset = IWSLT14Dataset(split="test",local_file="iwslt14_test.json")

# Debug #
train_dataset = IWSLT14Dataset(split="train",local_file="iwslt14_train_debug.json")
val_dataset = IWSLT14Dataset(split="validation",local_file="iwslt14_validation_debug.json")
test_dataset = IWSLT14Dataset(split="test",local_file="iwslt14_test_debug.json")

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Debug
src_vocab_size, trg_vocab_size = train_dataset.get_vocab_sizes()
max_length = train_dataset.get_max_length()

# Initialize the model #
st_model = SimpleTransformer(src_vocab_size, trg_vocab_size, embed_dim,
                             num_heads=num_heads, d_k=d_k, d_v=d_v).to(device)

# Loss & Optimization #
criterion = torch.nn.CrossEntropyLoss(ignore_index=train_dataset.get_padding_index()).to(device)
optimizer = optim.Adam(st_model.parameters(), lr=learning_rate)


# Training loop
def train() -> float:
    """
    Performs one epoch of training on the model.

    Returns:
        float: The average loss for the epoch.
    """
    st_model.train()
    total_loss = 0
    # start_time = time.time()

    for batch_idx, (src, trg) in enumerate(train_loader):
        # Move data to device (GPU/CPU)
        src, trg = src.to(device), trg.to(device)
        trg_shifted = utils.shift_trg_right(trg).to(device) # Remove the last token from target (teacher forcing)

        # Forward pass
        optimizer.zero_grad()
        output = st_model(src, trg_shifted)

        # Compute loss
        loss = criterion(output.view(-1, trg_vocab_size), trg.view(-1))  # Shift target by 1
        total_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Print loss every few steps
        # if batch_idx % 50 == 0:
        #     print(f"Batch {batch_idx}, Loss: {loss.item()}")

    avg_loss = total_loss / len(train_loader)
    # elapsed_time = time.time() - start_time
    print(f"Epoch complete, Average Train Loss: {avg_loss:.4f}") #(, Time: {elapsed_time:.2f} seconds)
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
        print(f"\nEpoch {epoch}/{epochs}")
        avg_loss = train()
        loss_record['train'].append(avg_loss)

        # Evaluate BLEU on the validation set
        val_loss = evaluation.evaluate_model(st_model, val_loader, criterion, device)
        print(f"Validation loss: {val_loss:.2f}")
        loss_record['validation'].append(val_loss)

        # Save the model if the loss is the best so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            utils.save_model(epoch, st_model, optimizer, avg_loss)

    return loss_record

# Entry point
if __name__ == "__main__":
    loss_records = train_model()
    test_loss = evaluation.evaluate_model(st_model, test_loader, criterion,  device)
    print(f"\nTest loss: {test_loss:.2f}")
    bleu_score = evaluation.evaluate_bleu(st_model, test_loader, test_dataset.fr_vocab,  device)
    print(f"\nBLEU on test set: {bleu_score:.2f}")
    utils.plot_losses(loss_records)


