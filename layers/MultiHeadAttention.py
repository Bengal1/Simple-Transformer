import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiHeadAttention(torch.nn.Module):
    """
    Multi-Head Attention module for Transformer architectures.

    Supports both self-attention and cross-attention mechanisms, with optional
    causal (autoregressive) masking. This module splits input embeddings across
    multiple attention heads, performs scaled dot-product attention in parallel,
    and then projects the result back to the original embedding space.
    """

    def __init__(self, embed_dim: int,
                 num_heads: int = 8,
                 d_k: int = 64,
                 d_v: int = 64,
                 dropout: float = 0.1,
                 cross_attn: bool = False,
                 masked_attn: bool = False):
        """Initializes the multi-head attention layer.

        Args:
            embed_dim (int): Total input and output embedding dimension.
            num_heads (int): Number of attention heads.
            d_k (int): Dimension of the query and key projections per head.
            d_v (int): Dimension of the value projection per head.
            dropout (float): Dropout probability applied to attention weights and output projection.
            cross_attn (bool): If True, enables cross-attention using a separate source input `y`.
            masked_attn (bool): If True, applies causal masking for autoregressive decoding.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.cross_attn = cross_attn
        self.masked_attn = masked_attn

        # Shared linear layers
        self.w_q = nn.Linear(embed_dim, num_heads * d_k)
        self.w_k = nn.Linear(embed_dim, num_heads * d_k)
        self.w_v = nn.Linear(embed_dim, num_heads * d_v)

        # Output projection
        self.w_out = nn.Linear(num_heads * d_v, embed_dim)
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        self.scale = 1.0 / math.sqrt(d_k)

        # Xavier initialization
        # init.xavier_uniform_(self.w_q.weight)
        # init.xavier_uniform_(self.w_k.weight)
        # init.xavier_uniform_(self.w_v.weight)
        # init.xavier_uniform_(self.w_out.weight)

    def _split_heads(self, x: torch.Tensor, head_dim: int) -> torch.Tensor:
        """Splits the last dimension into (num_heads, head_dim) and transposes to (B, H, L, D).

        Args:
            x (torch.Tensor): Tensor of shape (B, L, num_heads * head_dim).
            head_dim (int): The dimension size per attention head.

        Returns:
            torch.Tensor: Reshaped tensor of shape (B, num_heads, L, head_dim).
        """
        B, L, _ = x.size()
        return x.view(B, L, self.num_heads, head_dim).transpose(1, 2)

    @staticmethod
    def _combine_heads(x: torch.Tensor) -> torch.Tensor:
        """Combines the multi-head output into a single vector per position.

        Args:
            x (torch.Tensor): Tensor of shape (B, H, L, D).

        Returns:
            torch.Tensor: Reshaped tensor of shape (B, L, H * D).
        """
        B, H, L, D = x.size()
        return x.transpose(1, 2).contiguous().view(B, L, H * D)

    def _generate_causal_mask(self, L_q: int, L_k: int, device: torch.device) -> torch.Tensor:
        """
        Generates a causal (upper triangular) attention mask for autoregressive decoding.

        This mask prevents attention to future positions by setting the upper triangle
        (above the main diagonal) to negative infinity, which effectively masks those
        positions when added to the attention logits before softmax.

        Args:
            L_q (int): Length of the query sequence (usually the current input length).
            L_k (int): Length of the key sequence (memory size or same as L_q for self-attention).
            device (torch.device): Device on which to create the mask.

        Returns:
            torch.Tensor: A mask tensor of shape (1, 1, L_q, L_k), where masked positions
            contain -inf and others are 0. This shape supports broadcasting over batches
            and attention heads.
        """
        return torch.triu(torch.full((L_q, L_k), float('-inf'),device=device), diagonal=1)[None, None, :, :]

    def _scaled_dot_product_attention(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes scaled dot-product attention.

        Args:
            Q (torch.Tensor): Queries of shape (B, H, L_q, D).
            K (torch.Tensor): Keys of shape (B, H, L_k, D).
            V (torch.Tensor): Values of shape (B, H, L_k, D).
            mask (Optional[torch.Tensor]): Optional mask of shape (1, 1, L_q, L_k),
                where masked positions are set to -inf.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Output tensor of shape (B, H, L_q, D).
                - Attention weights of shape (B, H, L_q, L_k).
        """
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn_scores = attn_scores + mask  # Add -inf to masked positions

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        output = torch.matmul(attn_probs, V)
        return output, attn_probs

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for multi-head attention.

        Args:
            x: Tensor of shape (B, L_x, embed_dim)
            y: Optional tensor for cross-attention (B, L_y, embed_dim). If None, self-attention is used.
            return_attn_weights: If True, also returns attention weights.

        Returns:
            Output tensor of shape (B, L_x, embed_dim), and optionally attention weights.
                - Output tensor of shape (B, L_x, embed_dim)
                - Optionally, attention weights of shape (B, num_heads, L_x, L_y) if `return_attn_weights` is True.
        """
        L_x = x.shape[1]
        L_y = y.shape[1] if (self.cross_attn and y is not None) else L_x

        # Linear projections and reshape
        kv_source = x if not self.cross_attn or y is None else y
        Q = self._split_heads(self.w_q(x), self.d_k)
        K = self._split_heads(self.w_k(kv_source), self.d_k)
        V = self._split_heads(self.w_v(kv_source), self.d_v)

        mask = None
        if self.masked_attn:
            mask = self._generate_causal_mask(L_x, L_y, device=x.device)

        # Scaled dot-product attention
        attn_output, attn_weights = self._scaled_dot_product_attention(Q, K, V, mask)

        # Merge heads and project output
        merged = self._combine_heads(attn_output)
        output = self.out_dropout(self.w_out(merged))

        return (output, attn_weights) if return_attn_weights else output


# class MultiHeadAttention(torch.nn.Module):
#     """
#     Multi-Head Attention module for Transformer models.
#
#     This module implements the multi-head attention mechanism used in Transformer
#     architectures. It supports both self-attention and cross-attention, and can
#     optionally apply a causal mask for autoregressive decoding.
#
#     Attributes:
#         num_heads (int): Number of attention heads.
#         d_k (int): Dimension of key vectors per head.
#         d_v (int): Dimension of value vectors per head.
#         cross_attn (bool): Whether the module operates in cross-attention mode.
#         masked_attn (bool): Whether to apply a causal mask for decoding.
#         w_q (nn.ModuleList): List of linear layers for projecting query vectors.
#         w_k (nn.ModuleList): List of linear layers for projecting key vectors.
#         w_v (nn.ModuleList): List of linear layers for projecting value vectors.
#         w_out (nn.Linear): Output linear layer that projects concatenated attention outputs.
#         attn_dropout (nn.Dropout): Dropout applied to attention weights.
#         out_dropout (nn.Dropout): Dropout applied to the final output projection.
#     """
#
#     def __init__(self, embed_dim: int, num_heads: int = 1, d_k: int = 64,
#                  d_v: int = 128, dropout: float = 0.0, cross_attn: bool = False, masked_attn: bool = False):
#         """Initializes the MultiHeadAttention module.
#
#         Args:
#             embed_dim (int): Dimension of the input embeddings.
#             num_heads (int, optional): Number of attention heads. Defaults to 1.
#             d_k (int, optional): Dimension of the key vectors per head. Defaults to 64.
#             d_v (int, optional): Dimension of the value vectors per head. Defaults to 128.
#             dropout (float, optional): Dropout rate applied to attention weights and output. Defaults to 0.0.
#             cross_attn (bool, optional): If True, enables cross-attention mode for use in the decoder. Defaults to False.
#             masked_attn (bool, optional): If True, applies causal (masked) attention for autoregressive decoding. Defaults to False.
#         """
#         super().__init__()
#
#         self.num_heads = num_heads
#         self.d_k = d_k  # Size of key vectors per head
#         self.d_v = d_v  # Size of value vectors per head
#         self.cross_attn = cross_attn
#         self.masked_attn = masked_attn
#
#         # Linear projections for Q, K, V for each head
#         self.w_q = nn.ModuleList([nn.Linear(embed_dim, d_k) for _ in range(num_heads)])
#         self.w_k = nn.ModuleList([nn.Linear(embed_dim, d_k) for _ in range(num_heads)])
#         self.w_v = nn.ModuleList([nn.Linear(embed_dim, d_v) for _ in range(num_heads)])
#
#         self.w_out = nn.Linear(d_v * num_heads, embed_dim)
#         # Dropout
#         self.attn_dropout = nn.Dropout(dropout)
#         self.out_dropout = nn.Dropout(dropout)
#
#     def _scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
#                                       mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
#         """Computes scaled dot-product attention.
#
#         Args:
#             Q (torch.Tensor): Query tensor of shape (batch_size, num_heads, src_len, d_k).
#             K (torch.Tensor): Key tensor of shape (batch_size, num_heads, tgt_len, d_k).
#             V (torch.Tensor): Value tensor of shape (batch_size, num_heads, tgt_len, d_v).
#             mask (torch.Tensor, optional): Attention mask of shape (1, 1, src_len, tgt_len), with -inf for masked positions.
#
#         Returns:
#             tuple[torch.Tensor, torch.Tensor]:
#                 - Attention output of shape (batch_size, num_heads, src_len, d_v).
#                 - Attention probabilities of shape (batch_size, num_heads, src_len, tgt_len).
#         """
#         attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=Q.dtype, device=Q.device))
#
#         if mask is not None:
#             attn_scores = attn_scores + mask
#
#         attn_probs = F.softmax(attn_scores, dim=-1)
#         attn_probs = self.attn_dropout(attn_probs) # Dropout
#         return torch.matmul(attn_probs, V), attn_probs
#
#     def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
#         """Performs forward pass of multi-head attention.
#
#         Args:
#             x (torch.Tensor): Source tensor of shape (batch_size, src_len, embed_dim).
#             y (torch.Tensor, optional): Target tensor for cross-attention, shape (batch_size, tgt_len, embed_dim).
#                                         If None, self-attention is performed.
#
#         Returns:
#             torch.Tensor: Output tensor of shape (batch_size, src_len, embed_dim).
#         """
#         batch_size, src_len, _ = x.shape
#         _, trg_len, _ = y.shape if self.cross_attn else x.shape
#
#         # Initialize Q, K, V for each head
#         Q, K, V = [], [], []
#
#         for i in range(self.num_heads):
#             Q.append(self.w_q[i](x))
#             K.append(self.w_k[i](x if not self.cross_attn else y))
#             V.append(self.w_v[i](x if not self.cross_attn else y))
#
#         # Stack the Q, K, V tensors into one tensor of shape (batch_size, num_heads, max_length, d_k/d_v)
#         Q = torch.stack(Q, dim=1)
#         K = torch.stack(K, dim=1)
#         V = torch.stack(V, dim=1)
#
#         # Create dynamic mask of shape (1, 1, src_len, trg_len)
#         mask = None
#         if self.masked_attn:
#             mask = torch.triu(torch.full((src_len, trg_len), float('-inf'), device=x.device), diagonal=1)
#             mask = mask.unsqueeze(0).unsqueeze(1)  # (1, 1, src_len, trg_len)
#
#         # Apply scaled dot-product attention
#         attention_output, _ = self._scaled_dot_product_attention(Q, K, V, mask)
#
#         # Concatenate heads and project to output dimension + Dropout
#         attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, src_len, -1)
#         output = self.w_out(attention_output)
#         return self.out_dropout(output) # Dropout