# import os
# import time
# import torch
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from models import SimpleTransformer
# from datasets import IWSLT14Dataset
# import torch.nn.functional as F
#
# # Hyperparameters
# CONFIG = {
#     'embed_dim': 512,  # Embedding dimension
#     'max_length': 100,  # Maximum sequence length
#     'trg_vocab_size': 5000,  # Target vocabulary size (adjust based on dataset size)
#     'd_v': 10,  # Output dimension per head
#     'num_heads': 8,  # Number of attention heads
#     'batch_size': 32,  # Batch size
#     'epochs': 10,  # Number of epochs
#     'learning_rate': 1e-4  # Learning rate
# }
#
# # Device configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # Initialize dataset and DataLoader
# dataset = IWSLT14Dataset()
# train_loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
#
# # Model initialization
# model = SimpleTransformer(CONFIG['embed_dim'], CONFIG['max_length'], CONFIG['trg_vocab_size'], CONFIG['d_v'],
#                           CONFIG['num_heads']).to(device)
#
# # Loss function and optimizer
# criterion = torch.nn.CrossEntropyLoss(
#     ignore_index=dataset.get_padding_index())  # Ignore padding token in loss calculation
# optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
#
#
# # Utility function to save model checkpoint
# def save_checkpoint(epoch, model, optimizer, loss, filepath="model_checkpoint.pth"):
#     """
#     Save model checkpoint.
#
#     Args:
#         epoch (int): Current epoch number
#         model (nn.Module): Model to save
#         optimizer (torch.optim.Optimizer): Optimizer state to save
#         loss (float): Current loss value
#         filepath (str): Path to save the checkpoint
#     """
#     checkpoint = {
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'loss': loss
#     }
#     torch.save(checkpoint, filepath)
#     print(f"Checkpoint saved at epoch {epoch}. Loss: {loss:.4f}")
#
#
# # Training loop for a single epoch
# def train_one_epoch(epoch, model, train_loader, optimizer, criterion):
#     """
#     Train the model for one epoch.
#
#     Args:
#         epoch (int): Current epoch number
#         model (nn.Module): The model to train
#         train_loader (DataLoader): DataLoader for the training data
#         optimizer (torch.optim.Optimizer): Optimizer used for parameter updates
#         criterion (torch.nn.CrossEntropyLoss): Loss function
#     """
#     model.train()
#     total_loss = 0
#     start_time = time.time()
#
#     for batch_idx, (src, trg) in enumerate(train_loader):
#         src, trg = src.to(device), trg.to(device)
#
#         # Forward pass
#         optimizer.zero_grad()
#         output = model(src, trg[:, :-1])  # Teacher forcing: target is shifted by 1
#
#         # Calculate loss
#         loss = criterion(output.view(-1, CONFIG['trg_vocab_size']), trg[:, 1:].reshape(-1))  # Shift target by 1
#         total_loss += loss.item()
#
#         # Backward pass and optimization step
#         loss.backward()
#         optimizer.step()
#
#         # Log every 50 batches
#         if batch_idx % 50 == 0:
#             print(
#                 f"Epoch [{epoch + 1}/{CONFIG['epochs']}], Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
#
#     avg_loss = total_loss / len(train_loader)
#     epoch_time = time.time() - start_time
#     print(f"Epoch {epoch + 1} complete. Avg. Loss: {avg_loss:.4f}, Time taken: {epoch_time:.2f}s")
#     return avg_loss
#
#
# # Full training loop across multiple epochs
# def train_model(model, train_loader, optimizer, criterion, num_epochs=CONFIG['epochs']):
#     """
#     Train the model over multiple epochs.
#
#     Args:
#         model (nn.Module): Model to train
#         train_loader (DataLoader): DataLoader for training data
#         optimizer (torch.optim.Optimizer): Optimizer for training
#         criterion (torch.nn.CrossEntropyLoss): Loss function
#         num_epochs (int): Number of epochs to train for
#     """
#     best_loss = float('inf')
#
#     for epoch in range(num_epochs):
#         print(f"\nStarting Epoch {epoch + 1}/{num_epochs}")
#         avg_loss = train_one_epoch(epoch, model, train_loader, optimizer, criterion)
#
#         # Save the best model (lowest loss)
#         if avg_loss < best_loss:
#             best_loss = avg_loss
#             save_checkpoint(epoch, model, optimizer, avg_loss)
#
#
# # Main function to execute training
# def main():
#     """
#     Main entry point to start model training.
#     """
#     print("Training started...")
#     train_model(model, train_loader, optimizer, criterion)
#
#
# # Entry point
# if __name__ == "__main__":
#     main()
