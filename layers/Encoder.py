import torch
from typing import Optional
from .NormLayer import NormLayer
from .FeedForward import FeedForward
from .MultiHeadAttention import MultiHeadAttention


class Encoder(torch.nn.Module):
    """
    A single Transformer encoder block.

    This module represents a standard Transformer encoder block, which includes:
    - Multi-head self-attention
    - Layer normalization with residual connections
    - Position-wise feedforward network

    Attributes:
        attention (MultiHeadAttention): Multi-head self-attention mechanism.
        norm1 (NormLayer): Layer normalization after attention with residual connection.
        ff (FeedForward): Position-wise feedforward network.
        norm2 (NormLayer): Layer normalization after feedforward network with residual connection.
    """

    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 d_k: int,
                 d_v: int,
                 dropout: float = 0.1):
        """Initializes the Encoder block.

        Args:
            d_model (int): Dimensionality of the input embeddings.
            num_heads (int): Number of attention heads.
            d_k (int): Dimensionality of key vectors per head.
            d_v (int): Dimensionality of value vectors per head.
            dropout (float, optional): Dropout rate applied to attention and feedforward layers. Defaults to 0.0.
        """
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, d_k, d_v,
                                            dropout=dropout)
        # self.norm1 = NormLayer(d_model)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dropout=dropout)
        # self.norm2 = NormLayer(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(p=dropout)
        self.dropout2 = torch.nn.Dropout(p=dropout)

    def forward(self,
                enc_input: torch.Tensor,
                src_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Applies the encoder block forward pass.

        Args:
            enc_input (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            src_padding_mask (Optional[torch.Tensor], optional): Padding mask for the encoder input.
                Shape: (batch_size, seq_len). Positions with 1 are masked (ignored in attention). Default is None.
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        # Multi-head self-attention + residual + norm
        attn_out = self.attention(enc_input, padding_mask=src_padding_mask)
        norm1_out = self.norm1(self.dropout1(attn_out) + enc_input)

        # Feedforward network + residual + norm
        ff_out = self.ff(norm1_out)
        enc_out = self.norm2(self.dropout2(ff_out) + norm1_out)

        return enc_out