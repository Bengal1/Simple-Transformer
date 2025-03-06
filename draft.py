import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader

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
