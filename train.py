import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleTransformer

# Hyper Parameters #
learning_rate = 1e-3
num_epochs = 20
batch_size = 256
embed_dim_src =
embed_dim_target =
seq_len =
d_k =
d_v =

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using', device, '\n')

model = SimpleTransformer(embed_dim_src, embed_dim_target, seq_len, d_k, d_v)
model.to(device)

# Data #


# Loss & Optimization #
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train #


# Test #
model.eval()