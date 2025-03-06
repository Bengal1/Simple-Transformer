import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from model import SimpleTransformer
import utils

# Hyper Parameters #
learning_rate = 1e-3
num_epochs = 20
batch_size = 256
embed_dim = 265
seq_len = 20
file = "en_fr.csv"

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using', device, '\n')

# Data #
en_tokenized_sentence, fr_tokenized_sentence, max_length = utils.prepare_tokenized_vocabulary(file)


model = SimpleTransformer(embed_dim, max_length)
model.to(device)

# Loss & Optimization #
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train #


# Test #
model.eval()