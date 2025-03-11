import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.SimpeTransformer import SimpleTransformer
from data.iwslt14 import IWSLT14Dataset
import time


# Hyperparameters
embed_dim = 512          # Embedding dimension
max_length = 100         # Maximum sequence length
trg_vocab_size = 5000    # Target vocabulary size (adjust based on dataset size)
d_v = 10                 # Dimension for output
num_heads = 8            # Number of attention heads
batch_size = 32          # Batch size
epochs = 10              # Number of epochs
learning_rate = 1e-4     # Learning rate

# Set device #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using', device, '\n')

# Data #
dataset = IWSLT14Dataset()
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the model #
model = SimpleTransformer(embed_dim, max_length, trg_vocab_size, d_v, num_heads).to(device)

# Loss & Optimization #
criterion = torch.nn.CrossEntropyLoss(ignore_index=dataset.get_padding_index())  # Ignore padding token
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Function to save the model checkpoint
def save_model(epoch, model, optimizer, loss, path="model_checkpoint.pth"):
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
    model.train()
    total_loss = 0
    start_time = time.time()

    for batch_idx, (src, trg) in enumerate(train_loader):
        # Move data to device (GPU/CPU)
        src, trg = src.to(device), trg.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(src, trg[:, :-1])  # Remove the last token from target (teacher forcing)

        # Compute loss
        loss = criterion(output.view(-1, trg_vocab_size), trg[:, 1:].reshape(-1))  # Shift target by 1
        total_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Print loss every few steps
        if batch_idx % 50 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item()}")

    avg_loss = total_loss / len(train_loader)
    elapsed_time = time.time() - start_time
    print(f"Epoch complete, Average Loss: {avg_loss:.4f}, Time: {elapsed_time:.2f} seconds")
    return avg_loss

# Train #
def train_model():
    best_loss = float('inf')

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        avg_loss = train()

        # Save the model if the loss is the best so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(epoch, model, optimizer, avg_loss)

if __name__ == "__main__":
    train_model()
