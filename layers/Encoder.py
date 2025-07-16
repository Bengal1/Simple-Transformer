import torch
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
                 embed_dim: int,
                 num_heads: int,
                 d_k: int,
                 d_v: int,
                 dropout: float = 0.1):
        """Initializes the Encoder block.

        Args:
            embed_dim (int): Dimensionality of the input embeddings.
            num_heads (int): Number of attention heads.
            d_k (int): Dimensionality of key vectors per head.
            d_v (int): Dimensionality of value vectors per head.
            dropout (float, optional): Dropout rate applied to attention and feedforward layers. Defaults to 0.0.
        """
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, d_k, d_v,
                                            dropout=dropout)
        self.norm1 = NormLayer(embed_dim)

        self.ff = FeedForward(embed_dim, dropout=dropout)
        self.norm2 = NormLayer(embed_dim)

    def forward(self, enc_input: torch.Tensor) -> torch.Tensor:
        """Applies the encoder block forward pass.

        Args:
            enc_input (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        # Multi-head self-attention + residual + norm
        attn_out = self.attention(enc_input)
        norm1_out = self.norm1(attn_out + enc_input)

        # Feedforward network + residual + norm
        ff_out = self.ff(norm1_out)
        enc_out = self.norm2(ff_out + norm1_out)

        return enc_out