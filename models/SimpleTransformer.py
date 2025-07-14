"""
Simple Transformer Model in PyTorch.

This module implements a Transformer architecture for sequence-to-sequence
tasks, particularly for machine translation. The model consists of the
following core components:

- Encoder: A stack of multi-head attention layers, feed-forward networks, and
  layer normalization.
- Decoder: A stack of multi-head attention layers with an additional
  cross-attention mechanism that attends to the encoder's output.
- MultiHeadAttention: A custom implementation of multi-head attention,
  enabling the model to focus on different parts of the input sequence
  simultaneously.
- FeedForward: A position-wise feed-forward neural network that applies
  non-linearity after the attention layers.
- NormLayer: A layer normalization component to stabilize the training
  process.
- SimpleTransformer: The main Transformer model combining the encoder,
  decoder, and an output linear layer for sequence generation.
- PositionalEncoding: Adds positional information to the input sequence to
  help the model learn token order.

The `SimpleTransformer` model can be used for machine translation, text
generation, or other sequence-to-sequence tasks with appropriate tokenization
and loss functions. It supports various hyperparameters to control the depth
of the network, number of attention heads, and dimensionality of the model.

Modules:
  Encoder: Encodes the input sequence using self-attention and feed-forward
    networks.
  Decoder: Autoregressively generates the output sequence while attending to
    the encoder's output and previous tokens.
  MultiHeadAttention: Performs attention on the input sequence, allowing the
    model to focus on different parts in parallel.
  FeedForward: A feed-forward network that processes each token independently
    after the attention mechanism.
  NormLayer: Layer normalization applied at strategic points for training
    stability.
  PositionalEncoding: Adds information about the position of tokens in the
    sequence to the input embeddings.
  Dropout: Regularization is applied after key components (like attention
    layers) to reduce overfitting.

Usage:
    The `SimpleTransformer` class encapsulates the entire Transformer
    architecture. To train or evaluate the model, input sequences (tokenized)
    and output sequences must be provided. A suitable optimizer, loss
    function, and learning rate scheduler should be used for training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from typing import Optional, Tuple


# class PositionalEncoding(torch.nn.Module):
#     """
#     Adds positional encoding to the input tensor.
#
#     The positional encoding follows the formula from "Attention Is All You Need"
#     and helps the Transformer model retain positional information.
#
#     Attributes:
#         embed_dim (int): The embedding dimension of the model.
#         n (int): The base for the sinusoidal encoding.
#     """
#
#     def __init__(self, embed_dim: int, n: int = 10000):
#         """Initializes the positional encoding module.
#
#         Args:
#             embed_dim (int): The embedding dimension of the model.
#             n (int, optional): The base for the sinusoidal encoding. Default is 10000.
#         """
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.n = n
#
#     def _create_positional_encoding(self, seq_len: int, device: torch.device) -> torch.Tensor:
#         """Creates the positional encoding matrix.
#
#         The encoding is based on sinusoidal functions that encode relative
#         position information for each token.
#
#         Args:
#             seq_len (int): Length of the sequence for positional encoding.
#             device (torch.device): Device where the tensor should be allocated.
#
#         Returns:
#             torch.Tensor: The positional encoding matrix of shape (seq_len, embed_dim).
#         """
#         pos_encoding = torch.zeros(seq_len, self.embed_dim, device=device)
#         k_pos = torch.arange(seq_len, device=device).unsqueeze(dim=1).float()
#         _2i = torch.arange(0, self.embed_dim, step=2, device=device).float()
#
#         pos_encoding[:, 0::2] = torch.sin(k_pos / self.n ** (_2i / self.embed_dim))
#         pos_encoding[:, 1::2] = torch.cos(k_pos / self.n ** (_2i / self.embed_dim))
#
#         return pos_encoding
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Adds positional encoding to the input tensor.
#
#         Args:
#             x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
#
#         Returns:
#             torch.Tensor: Tensor with added positional encoding,
#             of shape (batch_size, seq_len, embed_dim).
#         """
#         batch_size, seq_len, _ = x.shape
#         pos_encoding = self._create_positional_encoding(seq_len, x.device)
#
#         return x + pos_encoding.unsqueeze(0)  # Broadcast across batch
#
#
# class FeedForward(torch.nn.Module):
#     """
#     Position-wise FeedForward neural network used in Transformer models.
#
#     This module applies two linear transformations with a ReLU activation and dropout
#     in between, as used in the original "Attention Is All You Need" paper.
#
#     Attributes:
#         fc1 (nn.Linear): The first linear layer that expands the input dimension to the hidden dimension.
#         fc2 (nn.Linear): The second linear layer that projects the hidden representation back to the original dimension.
#         mid_dropout (nn.Dropout): Dropout applied after the ReLU activation.
#         out_dropout (nn.Dropout): Dropout applied after the second linear layer.
#     """
#
#
#     def __init__(self, d_model: int, hidden_dim: int = 2048, dropout: float = 0.1):
#         """Initializes the FeedForward network.
#
#         Args:
#             d_model (int): The input and output feature dimension.
#             hidden_dim (int, optional): The hidden layer dimension. Default is 2048.
#             dropout (float, optional): The dropout probability. Default is 0.1.
#         """
#         super().__init__()
#         self.fc1 = nn.Linear(d_model, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, d_model)
#         self.mid_dropout = nn.Dropout(dropout)
#         self.out_dropout = nn.Dropout(dropout)
#
#         # Xavier initialization
#         # nn.init.xavier_uniform_(self.fc1.weight)
#         # nn.init.xavier_uniform_(self.fc2.weight)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Performs a forward pass through the FeedForward network.
#
#         The input tensor is passed through a linear layer, followed by ReLU activation,
#         dropout, and a final linear layer.
#
#         Args:
#             x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
#
#         Returns:
#             torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
#         """
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.mid_dropout(x)
#         x = self.fc2(x)
#         x = self.out_dropout(x)
#         return x
#
#
# class NormLayer(torch.nn.Module):
#     """
#     Implements layer normalization used in the Transformer.
#
#     This normalization technique stabilizes the training process by normalizing
#     inputs across the last dimension and scaling them with learnable parameters.
#
#     Attributes:
#         gamma (nn.Parameter): Learnable scale parameter initialized to ones.
#         beta (nn.Parameter): Learnable shift parameter initialized to zeros.
#         epsilon (float): A small value added to variance for numerical stability.
#     """
#
#     def __init__(self, d_model: int, epsilon: float = 1e-15):
#         """Initializes the layer normalization module.
#
#         Args:
#             d_model (int): The dimension of the input tensor.
#             epsilon (float, optional): A small value added to variance for numerical stability. Default is 1e-15.
#         """
#         super().__init__()
#         self.gamma = nn.Parameter(torch.ones(d_model))
#         self.beta = nn.Parameter(torch.zeros(d_model))
#         self.epsilon = epsilon
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Applies layer normalization to the input tensor.
#
#         Normalizes the input across the last dimension and applies learnable
#         scaling (`gamma`) and shifting (`beta`).
#
#         Args:
#             x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
#
#         Returns:
#             torch.Tensor: Normalized tensor of the same shape as the input.
#         """
#         mean = x.mean(dim=-1, keepdim=True)
#         var = x.var(dim=-1, unbiased=False, keepdim=True)
#         std = torch.sqrt(var + self.epsilon)  # cache sqrt for efficiency
#         normalized = (x - mean) / std
#         return self.gamma * normalized + self.beta
#
#
# class MultiHeadAttention(nn.Module):
#     """
#     Multi-Head Attention module for Transformer architectures.
#
#     Supports both self-attention and cross-attention mechanisms, with optional
#     causal (autoregressive) masking. This module splits input embeddings across
#     multiple attention heads, performs scaled dot-product attention in parallel,
#     and then projects the result back to the original embedding space.
#     """
#
#     def __init__(self, embed_dim: int,
#                  num_heads: int = 8,
#                  d_k: int = 64,
#                  d_v: int = 64,
#                  dropout: float = 0.1,
#                  cross_attn: bool = False,
#                  masked_attn: bool = False):
#         """Initializes the multi-head attention layer.
#
#         Args:
#             embed_dim (int): Total input and output embedding dimension.
#             num_heads (int): Number of attention heads.
#             d_k (int): Dimension of the query and key projections per head.
#             d_v (int): Dimension of the value projection per head.
#             dropout (float): Dropout probability applied to attention weights and output projection.
#             cross_attn (bool): If True, enables cross-attention using a separate source input `y`.
#             masked_attn (bool): If True, applies causal masking for autoregressive decoding.
#         """
#         super().__init__()
#
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.d_k = d_k
#         self.d_v = d_v
#         self.cross_attn = cross_attn
#         self.masked_attn = masked_attn
#
#         # Shared linear layers
#         self.w_q = nn.Linear(embed_dim, num_heads * d_k)
#         self.w_k = nn.Linear(embed_dim, num_heads * d_k)
#         self.w_v = nn.Linear(embed_dim, num_heads * d_v)
#
#         # Output projection
#         self.w_out = nn.Linear(num_heads * d_v, embed_dim)
#         # Dropout
#         self.attn_dropout = nn.Dropout(dropout)
#         self.out_dropout = nn.Dropout(dropout)
#
#         self.scale = 1.0 / math.sqrt(d_k)
#
#         # Xavier initialization
#         # init.xavier_uniform_(self.w_q.weight)
#         # init.xavier_uniform_(self.w_k.weight)
#         # init.xavier_uniform_(self.w_v.weight)
#         # init.xavier_uniform_(self.w_out.weight)
#
#     def _split_heads(self, x: torch.Tensor, head_dim: int) -> torch.Tensor:
#         """Splits the last dimension into (num_heads, head_dim) and transposes to (B, H, L, D).
#
#         Args:
#             x (torch.Tensor): Tensor of shape (B, L, num_heads * head_dim).
#             head_dim (int): The dimension size per attention head.
#
#         Returns:
#             torch.Tensor: Reshaped tensor of shape (B, num_heads, L, head_dim).
#         """
#         B, L, _ = x.size()
#         return x.view(B, L, self.num_heads, head_dim).transpose(1, 2)
#
#     @staticmethod
#     def _combine_heads(x: torch.Tensor) -> torch.Tensor:
#         """Combines the multi-head output into a single vector per position.
#
#         Args:
#             x (torch.Tensor): Tensor of shape (B, H, L, D).
#
#         Returns:
#             torch.Tensor: Reshaped tensor of shape (B, L, H * D).
#         """
#         B, H, L, D = x.size()
#         return x.transpose(1, 2).contiguous().view(B, L, H * D)
#
#     def _generate_causal_mask(self, L_q: int, L_k: int, device: torch.device) -> torch.Tensor:
#         """
#         Generates a causal (upper triangular) attention mask for autoregressive decoding.
#
#         This mask prevents attention to future positions by setting the upper triangle
#         (above the main diagonal) to negative infinity, which effectively masks those
#         positions when added to the attention logits before softmax.
#
#         Args:
#             L_q (int): Length of the query sequence (usually the current input length).
#             L_k (int): Length of the key sequence (memory size or same as L_q for self-attention).
#             device (torch.device): Device on which to create the mask.
#
#         Returns:
#             torch.Tensor: A mask tensor of shape (1, 1, L_q, L_k), where masked positions
#             contain -inf and others are 0. This shape supports broadcasting over batches
#             and attention heads.
#         """
#         return torch.triu(torch.full((L_q, L_k), float('-inf'),device=device), diagonal=1)[None, None, :, :]
#
#     def _scaled_dot_product_attention(
#         self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
#         mask: Optional[torch.Tensor] = None
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Computes scaled dot-product attention.
#
#         Args:
#             Q (torch.Tensor): Queries of shape (B, H, L_q, D).
#             K (torch.Tensor): Keys of shape (B, H, L_k, D).
#             V (torch.Tensor): Values of shape (B, H, L_k, D).
#             mask (Optional[torch.Tensor]): Optional mask of shape (1, 1, L_q, L_k),
#                 where masked positions are set to -inf.
#
#         Returns:
#             Tuple[torch.Tensor, torch.Tensor]:
#                 - Output tensor of shape (B, H, L_q, D).
#                 - Attention weights of shape (B, H, L_q, L_k).
#         """
#         attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
#
#         if mask is not None:
#             attn_scores = attn_scores + mask  # Add -inf to masked positions
#
#         attn_probs = F.softmax(attn_scores, dim=-1)
#         attn_probs = self.attn_dropout(attn_probs)
#
#         output = torch.matmul(attn_probs, V)
#         return output, attn_probs
#
#     def forward(
#         self, x: torch.Tensor, y: Optional[torch.Tensor] = None,
#         return_attn_weights: bool = False
#         ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
#         """Forward pass for multi-head attention.
#
#         Args:
#             x: Tensor of shape (B, L_x, embed_dim)
#             y: Optional tensor for cross-attention (B, L_y, embed_dim). If None, self-attention is used.
#             return_attn_weights: If True, also returns attention weights.
#
#         Returns:
#             Output tensor of shape (B, L_x, embed_dim), and optionally attention weights.
#                 - Output tensor of shape (B, L_x, embed_dim)
#                 - Optionally, attention weights of shape (B, num_heads, L_x, L_y) if `return_attn_weights` is True.
#         """
#         L_x = x.shape[1]
#         L_y = y.shape[1] if (self.cross_attn and y is not None) else L_x
#
#         # Linear projections and reshape
#         kv_source = x if not self.cross_attn or y is None else y
#         Q = self._split_heads(self.w_q(x), self.d_k)
#         K = self._split_heads(self.w_k(kv_source), self.d_k)
#         V = self._split_heads(self.w_v(kv_source), self.d_v)
#
#         mask = None
#         if self.masked_attn:
#             mask = self._generate_causal_mask(L_x, L_y, device=x.device)
#
#         # Scaled dot-product attention
#         attn_output, attn_weights = self._scaled_dot_product_attention(Q, K, V, mask)
#
#         # Merge heads and project output
#         merged = self._combine_heads(attn_output)
#         output = self.out_dropout(self.w_out(merged))
#
#         return (output, attn_weights) if return_attn_weights else output
#
#
# class Encoder(nn.Module):
#     """
#     A single Transformer encoder block.
#
#     This module represents a standard Transformer encoder block, which includes:
#     - Multi-head self-attention
#     - Layer normalization with residual connections
#     - Position-wise feedforward network
#
#     Attributes:
#         attention (MultiHeadAttention): Multi-head self-attention mechanism.
#         norm1 (NormLayer): Layer normalization after attention with residual connection.
#         ff (FeedForward): Position-wise feedforward network.
#         norm2 (NormLayer): Layer normalization after feedforward network with residual connection.
#     """
#
#     def __init__(self, embed_dim: int, num_heads: int, d_k: int, d_v: int, dropout: float = 0.1):
#         """Initializes the Encoder block.
#
#         Args:
#             embed_dim (int): Dimensionality of the input embeddings.
#             num_heads (int): Number of attention heads.
#             d_k (int): Dimensionality of key vectors per head.
#             d_v (int): Dimensionality of value vectors per head.
#             dropout (float, optional): Dropout rate applied to attention and feedforward layers. Defaults to 0.0.
#         """
#         super().__init__()
#         self.attention = MultiHeadAttention(embed_dim, num_heads, d_k, d_v, dropout=dropout)
#         self.norm1 = NormLayer(embed_dim)
#
#         self.ff = FeedForward(embed_dim, dropout=dropout)
#         self.norm2 = NormLayer(embed_dim)
#
#     def forward(self, enc_input: torch.Tensor) -> torch.Tensor:
#         """Applies the encoder block forward pass.
#
#         Args:
#             enc_input (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
#
#         Returns:
#             torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim).
#         """
#         # Multi-head self-attention + residual + norm
#         attn_out = self.attention(enc_input)
#         norm1_out = self.norm1(attn_out + enc_input)
#
#         # Feedforward network + residual + norm
#         ff_out = self.ff(norm1_out)
#         enc_out = self.norm2(ff_out + norm1_out)
#
#         return enc_out
#
#
# class Decoder(nn.Module):
#     """
#     A single Transformer decoder block.
#
#     This module represents a Transformer decoder block consisting of:
#     - Masked multi-head self-attention
#     - Multi-head cross-attention with encoder output
#     - Layer normalization with residual connections
#     - Position-wise feedforward network
#
#     Attributes:
#         attention_masked (MultiHeadAttention): Masked multi-head self-attention mechanism.
#         norm1 (NormLayer): Layer normalization after masked self-attention with residual connection.
#         attention_cross (MultiHeadAttention): Cross-attention mechanism using encoder output.
#         norm2 (NormLayer): Layer normalization after cross-attention with residual connection.
#         ff (FeedForward): Position-wise feedforward network.
#         norm3 (NormLayer): Layer normalization after feedforward network with residual connection.
#     """
#
#     def __init__(self, embed_dim: int, num_heads: int, d_k: int, d_v: int, dropout: float = 0.1):
#         """Initializes the Decoder block.
#
#         Args:
#             embed_dim (int): Dimensionality of the input embeddings.
#             num_heads (int): Number of attention heads.
#             d_k (int): Dimensionality of key vectors per head.
#             d_v (int): Dimensionality of value vectors per head.
#             dropout (float, optional): Dropout rate applied to attention and feedforward layers. Defaults to 0.1.
#         """
#         super().__init__()
#         self.attention_masked = MultiHeadAttention(embed_dim, num_heads, d_k, d_v, dropout=dropout, masked_attn=True)
#         self.norm1 = NormLayer(embed_dim)
#
#         self.attention_cross = MultiHeadAttention(embed_dim, num_heads, d_k, d_v, dropout=dropout, cross_attn=True)
#         self.norm2 = NormLayer(embed_dim)
#
#         self.ff = FeedForward(embed_dim, dropout=dropout)
#         self.norm3 = NormLayer(embed_dim)
#
#     def forward(self, dec_input: torch.Tensor, enc_output: torch.Tensor) -> torch.Tensor:
#         """Applies the decoder block forward pass.
#
#         Args:
#             dec_input (torch.Tensor): Decoder input tensor of shape (batch_size, trg_seq_len, embed_dim).
#             enc_output (torch.Tensor): Encoder output tensor of shape (batch_size, src_seq_len, embed_dim).
#
#         Returns:
#             torch.Tensor: Decoder output tensor of shape (batch_size, trg_seq_len, embed_dim).
#         """
#         # Masked self-attention + residual + norm
#         attn_masked = self.attention_masked(dec_input)
#         norm1 = self.norm1(attn_masked + dec_input)
#
#         # Cross-attention with encoder output + residual + norm
#         attn_cross = self.attention_cross(norm1, enc_output)
#         norm2 = self.norm2(attn_cross + norm1)
#
#         # Feedforward network + residual + norm
#         ff_out = self.ff(norm2)
#         dec_out = self.norm3(ff_out + norm2)
#
#         return dec_out


from layers.PositionalEncoding import PositionalEncoding
from layers.Encoder import Encoder
from layers.Decoder import Decoder


class SimpleTransformer(nn.Module):
    """
    A simplified Transformer model for sequence-to-sequence tasks such as translation.

    Attributes:
        embedding_encoder (nn.Embedding): Embedding layer for source tokens.
        embedding_decoder (nn.Embedding): Embedding layer for target tokens.
        positional_encoding_encoder (PositionalEncoding): Positional encoding for source input.
        positional_encoding_decoder (PositionalEncoding): Positional encoding for target input.
        dropout (nn.Dropout): Dropout applied after embeddings.
        encoder_layers (nn.ModuleList): Stacked encoder layers.
        decoder_layers (nn.ModuleList): Stacked decoder layers.
        w_o (nn.Linear): Final projection layer mapping decoder output to vocabulary logits.
        softmax (nn.Softmax): Softmax function applied to output logits.
    """
    def __init__(
            self,
            src_vocab_size: int,
            trg_vocab_size: int,
            embed_dim: int,
            num_heads: int = 8,
            num_layers: int = 6,
            d_k: int = 64,
            d_v: int = 64,
            dropout: float = 0.1):
        """Initializes the SimpleTransformer model.

        Args:
            src_vocab_size (int): Size of the source vocabulary.
            trg_vocab_size (int): Size of the target vocabulary.
            embed_dim (int): Dimensionality of token embeddings.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of encoder and decoder layers.
            d_k (int): Dimensionality of key vectors.
            d_v (int): Dimensionality of value vectors.
            dropout (float): Dropout rate.
        """
        super().__init__()

        # Token embeddings
        self.embedding_encoder = nn.Embedding(src_vocab_size, embed_dim, padding_idx=1)
        self.embedding_decoder = nn.Embedding(trg_vocab_size, embed_dim, padding_idx=1)

        # Positional encodings
        self.positional_encoding_encoder = PositionalEncoding(embed_dim)
        self.positional_encoding_decoder = PositionalEncoding(embed_dim)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Encoder and decoder stacks
        self.encoder_layers = nn.ModuleList([
            Encoder(embed_dim, num_heads, d_k, d_v, dropout) for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            Decoder(embed_dim, num_heads, d_k, d_v, dropout) for _ in range(num_layers)
        ])

        # Output projection
        self.w_o = nn.Linear(embed_dim, trg_vocab_size)
        self.softmax = nn.Softmax(dim=-1)

        # Xavier initialization - W_o
        # init.xavier_uniform_(self.w_o.weight)

        # Normal initialization - Embedding
        # init.normal_(self.embedding_encoder.weight, mean=0.0, std=1.0)
        # self.embedding_encoder.weight.data *= math.sqrt(embed_dim)
        # 
        # init.normal_(self.embedding_decoder.weight, mean=0.0, std=1.0)
        # self.embedding_decoder.weight.data *= math.sqrt(embed_dim)

    def forward(self, src: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Transformer model.

        Args:
            src (torch.Tensor): Source input tensor of shape (batch_size, src_seq_len).
            trg (torch.Tensor): Target input tensor of shape (batch_size, trg_seq_len).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, trg_seq_len, trg_vocab_size).
        """
        # Source embeddings + positional encoding
        src_embed = self.embedding_encoder(src)
        src_pe = self.positional_encoding_encoder(src_embed)
        src_pe = self.dropout(src_pe)

        # Target embeddings + positional encoding
        trg_embed = self.embedding_decoder(trg)
        trg_pe = self.positional_encoding_decoder(trg_embed)
        trg_pe = self.dropout(trg_pe)

        # Pass through stacked Encoders
        enc_output = src_pe
        for layer in self.encoder_layers:
            enc_output = layer(enc_output)

        # Pass through stacked Decoders
        dec_output = trg_pe
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output)

        # Output layer (no Softmax; handled by nn.CrossEntropyLoss)
        output = self.w_o(dec_output)
        return output

    def translate(self, src: torch.Tensor, beam_size: int = 2, max_len: int = None) -> torch.Tensor:
        """Translates source sequences using beam search decoding with length normalization.

        Args:
            src (torch.Tensor): Source tensor of shape (batch_size, src_seq_len).
            beam_size (int): Beam width for beam search.
            max_len (int): Maximum length of decoded sequences. If None, it's computed dynamically.

        Returns:
            torch.Tensor: Tensor of shape (batch_size, decoded_seq_len) with predicted token IDs.
        """
        if max_len is None or max_len <= 0:
            max_len = int(src.size(1) * 1.6) + 10

        with torch.no_grad():
            bos_token_id, eos_token_id, pad_token_id = 2, 3, 1
            batch_size = src.size(0)
            vocab_size = self.w_o.out_features
            device = src.device

            # === Encode the input ===
            src_embed = self.dropout(self.positional_encoding_encoder(self.embedding_encoder(src)))
            enc_output = src_embed
            for layer in self.encoder_layers:
                enc_output = layer(enc_output)

            # Repeat encoder output for each beam
            enc_output = enc_output.unsqueeze(1).repeat(1, beam_size, 1, 1)  # (B, beam, L, D)
            enc_output = enc_output.view(batch_size * beam_size, *enc_output.shape[2:])  # (B*beam, L, D)

            # === Initialize decoder inputs ===
            sequences = torch.full((batch_size * beam_size, 1), bos_token_id, dtype=torch.long, device=device)
            sequence_scores = torch.zeros(batch_size, beam_size, device=device)
            sequence_scores[:, 1:] = float('-inf')  # only keep 1st beam
            sequence_scores = sequence_scores.view(-1)  # (B*beam,)
            finished = torch.zeros_like(sequence_scores, dtype=torch.bool)

            alpha = 0.6  # Length penalty factor

            for _ in range(max_len):
                # Decoder embedding
                trg_embed = self.dropout(self.positional_encoding_decoder(self.embedding_decoder(sequences)))
                dec_output = trg_embed
                for layer in self.decoder_layers:
                    dec_output = layer(dec_output, enc_output)

                logits = self.w_o(dec_output[:, -1, :])  # Only last token logits
                log_probs = F.log_softmax(logits, dim=-1)

                # Prevent expansion of finished beams
                log_probs[finished] = float('-inf')
                log_probs[finished, eos_token_id] = 0  # allow only <eos>

                # Expand beams
                scores = sequence_scores.unsqueeze(1) + log_probs  # (B*beam, V)
                scores = scores.view(batch_size, -1)  # (B, beam * V)

                top_scores, top_indices = scores.topk(beam_size, dim=-1)  # select top beams
                beam_indices = top_indices // vocab_size
                token_indices = top_indices % vocab_size

                # Gather previous sequences
                batch_offset = torch.arange(batch_size, device=device).unsqueeze(1) * beam_size
                gather_indices = (beam_indices + batch_offset).view(-1)
                next_tokens = token_indices.view(-1)

                sequences = sequences[gather_indices]
                sequences = torch.cat([sequences, next_tokens.unsqueeze(1)], dim=-1)

                finished = finished[gather_indices]
                sequence_scores = top_scores.view(-1)

                # Update finished
                finished |= (next_tokens == eos_token_id)

                # === Length penalty ===
                lengths = sequences.new_full((batch_size * beam_size,), sequences.size(1), dtype=torch.float)
                eos_mask = sequences == eos_token_id
                has_eos = eos_mask.any(dim=1)
                eos_positions = eos_mask.float().argmax(dim=1)
                lengths[has_eos] = (eos_positions[has_eos] + 1).float()  # include <eos>

                length_penalty = ((5.0 + lengths) / 6.0).pow(alpha)
                normalized_scores = sequence_scores / length_penalty

                if finished.view(batch_size, beam_size).all(dim=1).all():
                    break

            # === Select best beam per batch ===
            normalized_scores = normalized_scores.view(batch_size, beam_size)
            best_beam = normalized_scores.argmax(dim=1)
            best_sequences = []

            for i in range(batch_size):
                idx = i * beam_size + best_beam[i]
                best_sequences.append(sequences[idx])

            return torch.nn.utils.rnn.pad_sequence(best_sequences, batch_first=True, padding_value=pad_token_id)


    # def translate(self, src: torch.Tensor, beam_size: int = 2, max_len: int = None) -> torch.Tensor:
    #     """Translates source sequences using beam search decoding with length normalization.
    #
    #     Args:
    #         src (torch.Tensor): Source tensor of shape (batch_size, src_seq_len).
    #         beam_size (int): Beam width for beam search.
    #         max_len (int): Maximum length of decoded sequences. If None, it's computed dynamically.
    #
    #     Returns:
    #         torch.Tensor: Tensor of shape (batch_size, decoded_seq_len) with predicted token IDs.
    #     """
    #     if max_len is None or max_len <= 0:
    #         max_len = int(src.size(1) * 1.6) + 10
    #
    #     with torch.no_grad():
    #         bos_token_id, eos_token_id, pad_token_id = 2, 3, 1
    #         batch_size = src.size(0)
    #         vocab_size = self.w_o.out_features
    #         device = src.device
    #
    #         # === Encode input ===
    #         src_embed = self.dropout(self.positional_encoding_encoder(self.embedding_encoder(src)))
    #         enc_output = src_embed
    #         for layer in self.encoder_layers:
    #             enc_output = layer(enc_output)
    #
    #         # Repeat encoder output for beam search
    #         enc_output = enc_output.unsqueeze(1).repeat(1, beam_size, 1, 1)
    #         enc_output = enc_output.view(batch_size * beam_size, *enc_output.shape[2:])  # (B*beam, src_len, dim)
    #
    #         # === Initialize beams ===
    #         sequences = torch.full((batch_size * beam_size, 1), bos_token_id, dtype=torch.long, device=device)
    #         sequence_scores = torch.zeros(batch_size, beam_size, device=device)
    #         sequence_scores[:, 1:] = float('-inf')  # Only keep the first beam alive initially
    #         sequence_scores = sequence_scores.view(-1)  # (B*beam,)
    #         finished = torch.zeros(batch_size * beam_size, dtype=torch.bool, device=device)
    #
    #         alpha = 0.6  # Length normalization factor
    #
    #         for _ in range(max_len):
    #             # Embed decoder input
    #             trg_embed = self.dropout(self.positional_encoding_decoder(self.embedding_decoder(sequences)))
    #             dec_output = trg_embed
    #             for layer in self.decoder_layers:
    #                 dec_output = layer(dec_output, enc_output)
    #
    #             logits = self.w_o(dec_output[:, -1, :])  # (B*beam, vocab_size)
    #             log_probs = F.log_softmax(logits, dim=-1)
    #
    #             # Prevent expansion of finished beams
    #             log_probs[finished] = float('-inf')
    #             log_probs[finished, eos_token_id] = 0  # Only allow <eos> for finished beams
    #
    #             # Compute new scores
    #             scores = sequence_scores.unsqueeze(1) + log_probs  # (B*beam, vocab_size)
    #             scores = scores.view(batch_size, -1)  # (B, beam*vocab)
    #
    #             # Select top-k
    #             top_scores, top_indices = scores.topk(beam_size, dim=-1)
    #             beam_indices = top_indices // vocab_size
    #             token_indices = top_indices % vocab_size
    #
    #             # Compute flat indices into sequences
    #             batch_offset = (torch.arange(batch_size, device=device) * beam_size).unsqueeze(1)
    #             gather_indices = (beam_indices + batch_offset).view(-1)
    #             next_tokens = token_indices.view(-1)
    #
    #             # Reorder and expand sequences
    #             sequences = sequences[gather_indices]
    #             sequences = torch.cat([sequences, next_tokens.unsqueeze(1)], dim=-1)
    #
    #             # Reorder finished and score states
    #             finished = finished[gather_indices]
    #             sequence_scores = top_scores.view(-1)
    #
    #             # Update finished beams
    #             finished |= (next_tokens == eos_token_id)
    #
    #             # Length normalization
    #             lengths = sequences.new_full((batch_size * beam_size,), sequences.size(1), dtype=torch.float)
    #             eos_mask = (sequences == eos_token_id)
    #             has_eos = eos_mask.any(dim=1)
    #             first_eos = eos_mask.float().argmax(dim=1)
    #             lengths[has_eos] = first_eos[has_eos].float() + 1  # Include the <eos> token
    #
    #             length_penalty = ((5.0 + lengths) / 6.0).pow(alpha)
    #             normalized_scores = sequence_scores / length_penalty
    #
    #             if finished.view(batch_size, beam_size).all(dim=1).all():
    #                 break
    #
    #         # Select best sequence per batch
    #         normalized_scores = normalized_scores.view(batch_size, beam_size)
    #         best_beam = normalized_scores.argmax(dim=1)
    #         best_sequences = []
    #
    #         for i in range(batch_size):
    #             idx = i * beam_size + best_beam[i]
    #             best_sequences.append(sequences[idx])
    #
    #         return torch.nn.utils.rnn.pad_sequence(best_sequences, batch_first=True, padding_value=pad_token_id)

    # def translate(self, src: torch.Tensor, beam_size: int = 2, max_len: int = None) -> torch.Tensor:
    #     """
    #     Translates source sequences using beam search decoding with length normalization.
    #
    #     Args:
    #         src (torch.Tensor): Source tensor of shape (batch_size, src_seq_len).
    #         beam_size (int): Beam width.
    #         max_len (int): Maximum length of decoded sequences.
    #
    #     Returns:
    #         torch.Tensor: Tensor of shape (batch_size, decoded_seq_len) with predicted token IDs.
    #     """
    #     if max_len <= 0:
    #         max_len = int(src.size(1) * 1.6) + 10
    #
    #     with torch.no_grad():
    #         pad_token_id, bos_token_id, eos_token_id = 1, 2, 3
    #         batch_size = src.size(0)
    #         vocab_size = self.w_o.out_features
    #         # === Encode input ===
    #         src_embed = self.dropout(self.positional_encoding_encoder(self.embedding_encoder(src)))
    #         enc_output = src_embed
    #         for layer in self.encoder_layers:
    #             enc_output = layer(enc_output)
    #
    #         # Repeat encoder output for beam search
    #         enc_output = enc_output.unsqueeze(1).repeat(1, beam_size, 1, 1)  # (batch, beam, src_len, dim)
    #         enc_output = enc_output.view(batch_size * beam_size, *enc_output.shape[2:])  # (batch*beam, src_len, dim)
    #
    #         # === Beam initialization ===
    #         sequences = torch.full((batch_size * beam_size, 1), bos_token_id, dtype=torch.long, device=src.device)
    #         sequence_scores = torch.zeros(batch_size, beam_size, device=src.device)
    #         sequence_scores[:, 1:] = float('-inf')  # Only keep first beam alive initially
    #         sequence_scores = sequence_scores.view(-1)  # (batch*beam,)
    #         finished = torch.zeros_like(sequence_scores, dtype=torch.bool)
    #
    #         alpha = 0.6  # length normalization coefficient
    #
    #         for _ in range(max_len):
    #             trg_embed = self.dropout(self.positional_encoding_decoder(self.embedding_decoder(sequences)))
    #             dec_output = trg_embed
    #             for layer in self.decoder_layers:
    #                 dec_output = layer(dec_output, enc_output)
    #
    #             logits = self.w_o(dec_output[:, -1, :])  # (batch*beam, vocab_size)
    #             log_probs = F.log_softmax(logits, dim=-1)
    #
    #             scores = sequence_scores.unsqueeze(1) + log_probs  # (batch*beam, vocab)
    #             scores[finished] = float('-inf')
    #             scores[finished, eos_token_id] = sequence_scores[finished]
    #
    #             # Select top-k scores
    #             scores = scores.view(batch_size, beam_size * vocab_size)
    #             top_scores, top_indices = scores.topk(beam_size, dim=-1)
    #
    #             beam_indices = top_indices // vocab_size
    #             token_indices = top_indices % vocab_size
    #
    #             new_sequences = []
    #             for i in range(batch_size):
    #                 for b in range(beam_size):
    #                     beam = beam_indices[i, b] + i * beam_size
    #                     token = token_indices[i, b].unsqueeze(0)
    #                     new_seq = torch.cat([sequences[beam], token], dim=0)
    #                     new_sequences.append(new_seq)
    #
    #             sequences = torch.stack(new_sequences, dim=0)
    #
    #             # === Apply Length Normalization ===
    #             lengths = torch.full((batch_size * beam_size,), sequences.size(1), dtype=torch.float, device=src.device)
    #             eos_mask = (sequences == eos_token_id)
    #             eos_pos = eos_mask.float().argmax(dim=1)  # First <eos> position
    #             has_eos = eos_mask.any(dim=1)
    #             lengths[has_eos] = eos_pos[has_eos].float() + 1  # Add 1 to include the <eos> token
    #
    #             length_penalty = ((5.0 + lengths) ** alpha) / ((5.0 + 1.0) ** alpha)
    #             sequence_scores = top_scores.view(-1) / length_penalty
    #
    #             # Track finished beams
    #             finished |= (sequences[:, -1] == eos_token_id)
    #             if finished.view(batch_size, beam_size).all(dim=1).all():
    #                 break
    #
    #         # Select best sequence for each batch
    #         sequence_scores = sequence_scores.view(batch_size, beam_size)
    #         best_beam = sequence_scores.argmax(dim=-1)
    #         best_sequences = []
    #         for i in range(batch_size):
    #             best_idx = i * beam_size + best_beam[i]
    #             best_sequences.append(sequences[best_idx])
    #
    #         return torch.nn.utils.rnn.pad_sequence(best_sequences, batch_first=True, padding_value=pad_token_id)
