"""
This module implements the Multi-Head Attention mechanism, a core component
of the Transformer architecture. It supports both self-attention and
cross-attention, with an optional causal mask for autoregressive decoding
in sequence generation tasks.

The module provides:
- `MultiHeadAttention` class: A PyTorch module for performing multi-head
  scaled dot-product attention.
- Helper methods for splitting and combining attention heads, and generating
  causal masks.
- Integration of dropout for regularization.
- Support for applying padding masks to attention scores.
"""
import math
import torch
import torch.nn.functional as F
from typing import Optional


class MultiHeadAttention(torch.nn.Module):
    """
    Multi-Head Attention module for Transformer architectures.

    Supports both self-attention and cross-attention mechanisms, with optional
    causal (autoregressive) masking. This module splits input embeddings across
    multiple attention heads, performs scaled dot-product attention in parallel,
    and then projects the result back to the original embedding space.
    """

    def __init__(self, d_model: int,
                 num_heads: int = 8,
                 d_k: int = 64,
                 d_v: int = 64,
                 dropout: float = 0.1,
                 cross_attn: bool = False,
                 masked_attn: bool = False):
        """Initializes the multi-head attention layer.

        Args:
            d_model (int): Input and output embedding dimension.
            num_heads (int): Number of attention heads.
            d_k (int): Dimension of the query and key projections per head.
            d_v (int): Dimension of the value projection per head.
            dropout (float): Dropout probability applied to attention weights and output projection.
            cross_attn (bool): If True, enables cross-attention using a separate source input `y`.
            masked_attn (bool): If True, applies causal masking for autoregressive decoding.
        """
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.cross_attn = cross_attn
        self.masked_attn = masked_attn

        # Shared linear layers
        self.w_q = torch.nn.Linear(d_model, num_heads * d_k, bias=False)
        self.w_k = torch.nn.Linear(d_model, num_heads * d_k, bias=False)
        self.w_v = torch.nn.Linear(d_model, num_heads * d_v, bias=False)

        # Output projection
        self.w_out = torch.nn.Linear(num_heads * d_v, d_model, bias=False)

        # Dropout
        self.dropout = torch.nn.Dropout(dropout)

        self.scale = 1.0 / math.sqrt(d_k)

        # Xavier initialization
        torch.nn.init.xavier_uniform_(self.w_q.weight)
        torch.nn.init.xavier_uniform_(self.w_k.weight)
        torch.nn.init.xavier_uniform_(self.w_v.weight)
        torch.nn.init.xavier_uniform_(self.w_out.weight)

    def _split_heads(self, x: torch.Tensor, head_dim: int) -> torch.Tensor:
        """Splits the last dimension into (num_heads, head_dim) and
            transposes to (batch_size, head_num, sqe_len, head_dim).

        Args:
            x (torch.Tensor): Tensor of shape (batch_size, seq_len, num_heads * head_dim).
            head_dim (int): The dimension size per attention head.

        Returns:
            torch.Tensor: Reshaped tensor of shape
                            (batch_size, num_heads, seq_len, head_dim).
        """
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads,
                      head_dim).transpose(1, 2)

    @staticmethod
    def _combine_heads(x: torch.Tensor) -> torch.Tensor:
        """Combines the multi-head output into a single vector per position.

        Args:
            x (torch.Tensor): Tensor of shape (batch_size, head_num, seq_len, head_dim).

        Returns:
            torch.Tensor: Reshaped tensor of shape (batch_size, seq_len, head_num * head_dim).
        """
        batch_size, head_num, seq_len, head_dim = x.size()
        return x.transpose(1, 2).contiguous().view(
                batch_size, seq_len, head_num * head_dim)

    @staticmethod
    def _generate_causal_mask(
                              L_q: int,
                              L_k: int,
                              device: torch.device) -> torch.Tensor:
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
        return torch.triu(torch.full((L_q, L_k), float('-inf'),device=device),
                          diagonal=1)[None, None, :, :]

    def _scaled_dot_product_attention(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes scaled dot-product attention.

        Args:
            Q (torch.Tensor): Queries of shape (B, H, L_q, D).
            K (torch.Tensor): Keys of shape (B, H, L_k, D).
            V (torch.Tensor): Values of shape (B, H, L_k, D).
            mask (Optional[torch.Tensor]): Optional mask of shape (1, 1, L_q, L_k),
                where masked positions are set to -inf.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Output tensor of shape (B, H, L_q, D).
                - Attention weights of shape (B, H, L_q, L_k).
        """
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn_scores = attn_scores + mask  # Add -inf to masked positions

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        output = torch.matmul(attn_probs, V)
        return output, attn_probs

    def forward(self,
                x: torch.Tensor,
                y: Optional[torch.Tensor] = None,
                padding_mask: Optional[torch.Tensor] = None,
                return_attn_weights: bool = False
                ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for multi-head attention.

        Args:
            x: Tensor of shape (B, L_x, d_model)
            y: Optional tensor for cross-attention (batch_size, L_y, d_model). If None, self-attention is used.
            padding_mask (Optional[torch.Tensor], optional): Padding mask applied to the attention scores.
                Shape: (batch_size, L_y) for cross-attention or (batch_size, L_x) for self-attention. Default is None.
            return_attn_weights: If True, also returns attention weights. Default is False.

        Returns:
            Output tensor of shape (batch_size, L_x, d_model), and optionally attention weights.
                - Output tensor of shape (batch_size, L_x, d_model)
                - Optionally, attention weights of shape (batch_size, num_heads,
                                        L_x, L_y) if `return_attn_weights` is True.
        """
        if self.cross_attn and y is not None:
            kv_source = y
            L_x = x.shape[1]
            L_y = y.shape[1]
        else: # Self-attention
            kv_source = x
            L_x = x.shape[1]
            L_y = L_x

        # Linear projections and reshape
        Q = self._split_heads(self.w_q(x), self.d_k)
        K = self._split_heads(self.w_k(kv_source), self.d_k)
        V = self._split_heads(self.w_v(kv_source), self.d_v)

        combined_mask = None
        if self.masked_attn:
            combined_mask = self._generate_causal_mask(L_x, L_y, device=x.device)

        if padding_mask is not None:
            padding_mask_float = torch.zeros_like(padding_mask,
                                                  dtype=x.dtype).masked_fill_(
                                                    padding_mask, float('-inf'))
            if combined_mask is None:
                combined_mask = padding_mask_float
            else:
                combined_mask = torch.min(combined_mask, padding_mask_float)

        # Scaled dot-product attention
        attn_output, attn_weights = self._scaled_dot_product_attention(
                                                            Q, K, V, combined_mask)

        # Merge heads and project output
        merged = self._combine_heads(attn_output)
        # output = self.dropout(self.w_out(merged))
        output = self.w_out(merged)

        return (output, attn_weights) if return_attn_weights else output

