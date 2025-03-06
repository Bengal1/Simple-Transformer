import torch
import torch.nn as nn
import torchtext
import pandas as pd
from torchtext.data.utils import get_tokenizer
from torch.utils.data import Dataset, DataLoader

from torchtext.vocab import build_vocab_from_iterator
from collections import Counter

"""
--- Self-Attention ---
Input: X ∈ M×N

X·W_q = Q ∈ M×d - query matrix
X·W_k = K ∈ M×d - key matrix
X·W_v = V ∈ M×d_v - value matrix

Attention(Q,K,V) =Softmax(Q·K^T/sqrt(d))·V = ΔX' ∈ M×d_v
ΔX'·W_out = ΔX ∈ M×N
--> Z = ΔX + X - residual connection

--- Cross-Attention ---
Input: X ∈ M×N, Y ∈ L×N

X·W_q = Q ∈ M×d - query matrix
Y·W_k = K ∈ L×d - key matrix
Y·W_v = V ∈ L×d_v - value matrix

Attention(Q,K,V) =Softmax(Q·K^T/sqrt(d))·V = ΔX ∈ M×d_v
ΔX'·W_out = ΔX ∈ M×N
--> Z = ΔX + X - residual connection
"""

# ----------------------------------------------------------------------------------------------------

class TranslationDataset(Dataset):
    def __init__(self, src_sentences, trg_sentences, src_vocab, trg_vocab, max_len, src_embedding, trg_embedding):
        self.src_sentences = src_sentences
        self.trg_sentences = trg_sentences
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.max_len = max_len
        self.src_embedding = src_embedding
        self.trg_embedding = trg_embedding

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        # Add <bos> and <eos> tokens to the sentence
        src_sentence = [self.src_vocab['<bos>']] + [self.src_vocab[token] for token in self.src_sentences[idx]] + [self.src_vocab['<eos>']]
        trg_sentence = [self.trg_vocab['<bos>']] + [self.trg_vocab[token] for token in self.trg_sentences[idx]] + [self.trg_vocab['<eos>']]

        # Pad the sentences if they are shorter than max_length
        src_sentence = src_sentence + [self.src_vocab['<pad>']] * (self.max_len - len(src_sentence))
        trg_sentence = trg_sentence + [self.trg_vocab['<pad>']] * (self.max_len - len(trg_sentence))

        # Convert to tensors
        src_tensor = torch.tensor(src_sentence)
        trg_tensor = torch.tensor(trg_sentence)

        # Get embeddings for the token IDs
        src_embedded = self.src_embedding(src_tensor)  # (max_len, embed_dim)
        trg_embedded = self.trg_embedding(trg_tensor)  # (max_len, embed_dim)

        return src_embedded, trg_embedded


# Step 1: Read and preprocess data
df = pd.read_csv("en_fr.csv")

# Define tokenizers
en_tokenizer = get_tokenizer('spacy', language='en')
fr_tokenizer = get_tokenizer('spacy', language='fr')

# Tokenize the English and French sentences
df['tokenized_en'] = df['english'].apply(en_tokenizer)
df['tokenized_fr'] = df['french'].apply(fr_tokenizer)

# Define special tokens
special_tokens = ['<unk>', '<pad>', '<bos>', '<eos>']

# Build vocabulary for source (English) and target (French)
en_vocab = torchtext.vocab.build_vocab_from_iterator(df['tokenized_en'], specials=special_tokens)
fr_vocab = torchtext.vocab.build_vocab_from_iterator(df['tokenized_fr'], specials=special_tokens)

# Set default token for unknown words
en_vocab.set_default_index(en_vocab['<unk>'])
fr_vocab.set_default_index(fr_vocab['<unk>'])

# Define embedding dimensions
embed_dim = 256  # You can change this value

# Define the maximum sentence length
max_length = max(df['tokenized_en'].apply(len).max(), df['tokenized_fr'].apply(len).max()) + 2  # Adding 2 for <bos> and <eos>

# Step 2: Dataset Class for DataLoader with Embedding


# Step 3: Create Embedding layers
src_embedding = nn.Embedding(len(en_vocab), embed_dim)
trg_embedding = nn.Embedding(len(fr_vocab), embed_dim)

# Step 4: Create the Dataset and DataLoader
dataset = TranslationDataset(df['tokenized_en'], df['tokenized_fr'], en_vocab, fr_vocab, max_length, src_embedding, trg_embedding)

# Create a DataLoader (batch size can be changed as needed)
batch_size = 32
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Example: Iterate over DataLoader
for src_batch, trg_batch in data_loader:
    print(f"Source batch shape: {src_batch.shape}")  # (batch_size, max_len, embed_dim)
    print(f"Target batch shape: {trg_batch.shape}")  # (batch_size, max_len, embed_dim)
    break  # We just want to see the output for the first batch


# ---------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchtext
from torchtext.data.utils import get_tokenizer

# Step 1: Read and preprocess data
df = pd.read_csv("en_fr.csv")

# Define tokenizers
en_tokenizer = get_tokenizer('spacy', language='en')
fr_tokenizer = get_tokenizer('spacy', language='fr')

# Tokenize the English and French sentences
df['tokenized_en'] = df['english'].apply(en_tokenizer)
df['tokenized_fr'] = df['french'].apply(fr_tokenizer)

# Define special tokens
special_tokens = ['<unk>', '<pad>', '<bos>', '<eos>']

# Build vocabulary for source (English) and target (French)
en_vocab = torchtext.vocab.build_vocab_from_iterator(df['tokenized_en'], specials=special_tokens)
fr_vocab = torchtext.vocab.build_vocab_from_iterator(df['tokenized_fr'], specials=special_tokens)

# Set default token for unknown words
en_vocab.set_default_index(en_vocab['<unk>'])
fr_vocab.set_default_index(fr_vocab['<unk>'])

# Define the maximum sentence length
max_length = max(df['tokenized_en'].apply(len).max(),
                 df['tokenized_fr'].apply(len).max()) + 2  # Adding 2 for <bos> and <eos>


# Step 2: Dataset Class (No Embedding inside Dataset!)
class TranslationDataset(Dataset):
    def __init__(self, src_sentences, trg_sentences, max_len):
        self.src_sentences = src_sentences
        self.trg_sentences = trg_sentences
        self.max_len = max_len

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        # Add <bos> and <eos> tokens to the sentence
        src_sentence = ['<bos>'] + self.src_sentences[idx] + ['<eos>']
        trg_sentence = ['<bos>'] + self.trg_sentences[idx] + ['<eos>']

        # Pad the sentences if they are shorter than max_length
        src_sentence = src_sentence + ['<pad>'] * (self.max_len - len(src_sentence))
        trg_sentence = trg_sentence + ['<pad>'] * (self.max_len - len(trg_sentence))

        return src_sentence, trg_sentence  # Returning tokenized text (not IDs, not embeddings!)


# Step 3: Create the Dataset and DataLoader
dataset = TranslationDataset(df['tokenized_en'], df['tokenized_fr'], max_length)

batch_size = 32
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Step 4: Embedding Inside the Model (Not in Dataset!)
class TransformerModel(nn.Module):
    def __init__(self, vocab_size_src, vocab_size_trg, embed_dim):
        super(TransformerModel, self).__init__()
        self.src_embedding = nn.Embedding(vocab_size_src, embed_dim)
        self.trg_embedding = nn.Embedding(vocab_size_trg, embed_dim)

    def forward(self, src_tokens, trg_tokens, src_vocab, trg_vocab):
        # Convert tokens to indices inside the model!
        src_indices = torch.tensor([[src_vocab[token] for token in sentence] for sentence in src_tokens])
        trg_indices = torch.tensor([[trg_vocab[token] for token in sentence] for sentence in trg_tokens])

        # Convert token IDs to embeddings
        src_embedded = self.src_embedding(src_indices)
        trg_embedded = self.trg_embedding(trg_indices)

        return src_embedded, trg_embedded


# Step 5: Initialize Model
embed_dim = 256
model = TransformerModel(len(en_vocab), len(fr_vocab), embed_dim)

# Example usage
for src_tokens, trg_tokens in data_loader:
    src_emb, trg_emb = model(src_tokens, trg_tokens, en_vocab, fr_vocab)
    print("Source embeddings shape:", src_emb.shape)
    print("Target embeddings shape:", trg_emb.shape)
    break




# class MultiHeadAttention(nn.Module):
#     def __init__(self, max_length, embed_dim, num_heads=8,
#                  cross_attn=False, masked_attn=False):
#         super(MultiHeadAttention, self).__init__()
#
#         assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
#
#         self.cross_attn = cross_attn
#         self.masked_attn = masked_attn
#         self.num_heads = num_heads
#         self.d_k = embed_dim // num_heads  # Dimension per head
#
#         # Linear projections for Q, K, V
#         self.w_q = nn.Linear(embed_dim, embed_dim)
#         self.w_k = nn.Linear(embed_dim, embed_dim)
#         self.w_v = nn.Linear(embed_dim, embed_dim)
#         self.w_out = nn.Linear(embed_dim, embed_dim)
#
#         # Mask for decoder's self-attention
#         if masked_attn:
#             mask = torch.triu(torch.ones(max_length, max_length), diagonal=1)  # Upper triangular mask
#             self.register_buffer("mask", mask)
#
#     @staticmethod
#     def scaled_dot_product_attention(self, Q, K, V, mask=None):
#         """
#         Q, K, V are of shape (batch_size, num_heads, max_length, d_k)
#         mask is of shape (1, 1, max_length, max_length) (broadcastable)
#         """
#         d_k = Q.size(-1)
#         attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
#
#         if mask is not None:
#             attn_scores = attn_scores.masked_fill(mask, float('-inf'))
#
#         attn_probs = F.softmax(attn_scores, dim=-1)
#         return torch.matmul(attn_probs, V), attn_probs  # (batch_size, num_heads, max_length, d_k)
#
#     def forward(self, x, y=None):
#         """
#         x: (batch_size, max_length, embed_dim)  - always used
#         y: (batch_size, max_length, embed_dim)  - used in cross-attention (encoder-decoder)
#         """
#
#         batch_size, max_length, _ = x.shape
#
#         # If cross attention, y is the second input
#         if self.cross_attn and y is not None:
#             K, V = self.w_k(y), self.w_v(y)
#         else:
#             K, V = self.w_k(x), self.w_v(x)
#
#         Q = self.w_q(x)
#
#         # Reshape to (batch_size, num_heads, max_length, d_k)
#         Q = Q.view(batch_size, max_length, self.num_heads, self.d_k).transpose(1, 2)
#         K = K.view(batch_size, max_length, self.num_heads, self.d_k).transpose(1, 2)
#         V = V.view(batch_size, max_length, self.num_heads, self.d_k).transpose(1, 2)
#
#         # Apply attention with optional masking
#         mask = self.mask.unsqueeze(0).unsqueeze(0) if self.masked_attn else None
#         attention_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)
#
#         # Concatenate heads and project
#         attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, max_length, -1)
#         return self.w_out(attention_output)  # Final projection


# class MultiHeadAttention(nn.Module):
#     """
#     --- Self-Attention ---
#     Input: X ∈ M×E
#
#     X·W_q = Q ∈ M×d - query matrix
#     X·W_k = K ∈ M×d - key matrix
#     X·W_v = V ∈ M×d_v - value matrix
#
#     Attention(Q,K,V) =Softmax(Q·K^T/sqrt(d))·V = ΔX' ∈ M×d_v
#     ΔX'·W_out = ΔX ∈ M×E
#     --> Z = ΔX + X - residual connection
#
#     --- Cross-Attention ---
#     Input: X ∈ M×N, Y ∈ L×E
#
#     X·W_q = Q ∈ M×d - query matrix
#     Y·W_k = K ∈ L×d - key matrix
#     Y·W_v = V ∈ L×d_v - value matrix
#
#     Attention(Q,K,V) =Softmax(Q·K^T/sqrt(d))·V = ΔX ∈ M×d_v
#     ΔX'·W_out = ΔX ∈ M×E
#     --> Z = ΔX + X - residual connection
#     """
#
#     def __init__(self, max_length, d_k, d_v, num_heads, embed_dim, cross_attn=False,
#                  batch_first=True, attn_mask=False):
#         super(MultiHeadAttention, self).__init__()
#         self.cross_attn = cross_attn
#         self.attn_mask = attn_mask
#         if attn_mask:
#             arr = torch.new_full((max_length, embed_dim), fill_value=1e10, dtype=torch.float64)
#             self.mask = torch.triu(arr, diagonal=1)
#
#         self.w_q = nn.Linear(embed_dim, d_k)
#         self.w_k = nn.Linear(embed_dim, d_k)
#         self.w_v = nn.Linear(embed_dim, d_v)
#         self.w_out = nn.Linear(d_v, embed_dim)
#         self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=batch_first)
#
#
#     def scaled_dot_product_attention(self, Q, K, V, mask=None):
#         """
#         Q, K, V are of shape (batch_size, num_heads, max_length, d_k)
#         mask is of shape (1, 1, max_length, max_length) (broadcastable)
#         """
#         d_k = Q.size(-1)
#         attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
#
#         if mask is not None:
#             attn_scores = attn_scores.masked_fill(mask == 1, float('-inf'))
#
#         attn_probs = F.softmax(attn_scores, dim=-1)
#         return torch.matmul(attn_probs, V), attn_probs  # (batch_size, num_heads, max_length, d_k)
#
#
#     def forward(self, x, **args):
#         query = self.w_q(x)
#         if self.cross_attn:
#             y = args['y']
#             key = self.w_k(y)
#             value = self.w_v(y)
#         else:
#             key = self.w_k(x)
#             value = self.w_v(x)
#
#         if self.attn_mask:
#             delta_x_ = self.attention(query,key,value, attn_mask=self.mask)
#         else:
#             delta_x_ = self.attention(query, key, value)
#
#         delta_x = self.w_out(delta_x_)
#
#         return x + delta_x
