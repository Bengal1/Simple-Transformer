import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.SimpeTransformer import SimpleTransformer
from data.iwslt14 import IWSLT14Dataset
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
# dataset = IWSLT14Dataset()
dataset = IWSLT14Dataset(local_file="iwslt14_full.json")  # debug code!!!

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
src_vocab_size, trg_vocab_size = dataset.get_vocab_sizes()
max_length = dataset.get_max_length()
print(f"max_length = {max_length}, src_vocab_size = {src_vocab_size}, trg_vocab_size = {trg_vocab_size}") # debug code!!!

# Initialize the model #
st_model = SimpleTransformer(src_vocab_size, trg_vocab_size, embed_dim, max_length,
                             num_heads=num_heads, d_k=d_k, d_v=d_v).to(device)

# Loss & Optimization #
criterion = torch.nn.CrossEntropyLoss(ignore_index=dataset.get_padding_index())  # Ignore padding token
optimizer = optim.Adam(st_model.parameters(), lr=learning_rate)


def shift_trg_right(batch, eos_token_idx=3, pad_token_idx=1):
    """
    Convert <eos> token to <pad> token for each sentence in a batch of tensors.
    Designed for right-shift the target sequence in training transformer

    Parameters:
    - batch (Tensor): A batch of sentences (shape: [batch_size, seq_len]).
    - eos_token_idx (int): The index of the <eos> token.
    - pad_token_idx (int): The index of the <pad> token.

    Returns:
    - Tensor: The batch with <eos> replaced by <pad> for each sentence.
    """
    # Replace eos_token_idx with pad_token_idx in the batch
    batch[batch == eos_token_idx] = pad_token_idx
    return batch

# Function to save the model checkpoint
def save_model(epoch, model, optimizer, loss, path="model_checkpoint.pth"):
    """
    Save model checkpoint.

    Args:
        epoch (int): Current epoch number
        model (nn.Module): Model to save
        optimizer (torch.optim.Optimizer): Optimizer state to save
        loss (float): Current loss value
        filepath (str): Path to save the checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, path)
    print(f"Model checkpoint saved at epoch {epoch}.")

# Training loop
def train():
    st_model.train()
    total_loss = 0
    # start_time = time.time()

    for batch_idx, (src, trg) in enumerate(train_loader):
        # Move data to device (GPU/CPU)
        src, trg = src.to(device), trg.to(device)
        trg_shifted = shift_trg_right(trg)

        # Forward pass
        optimizer.zero_grad()
        output = st_model(src, trg_shifted)  # Remove the last token from target (teacher forcing)

        # Compute loss
        loss = criterion(output.view(-1, trg_vocab_size), trg.view(-1))  # Shift target by 1
        total_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Print loss every few steps
        if batch_idx % 50 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item()}")

    avg_loss = total_loss / len(train_loader)
    # elapsed_time = time.time() - start_time
    print(f"Epoch complete, Average Loss: {avg_loss:.4f}") #, Time: {elapsed_time:.2f} seconds")
    return avg_loss

# Train #
def train_model():
    best_loss = float('inf')

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        avg_loss = train()

        # Save the model if the loss is the best so far
        # if avg_loss < best_loss:
        #     best_loss = avg_loss
        #     save_model(epoch, st_model, optimizer, avg_loss)

if __name__ == "__main__":
    train_model()
