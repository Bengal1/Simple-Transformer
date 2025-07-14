import torch
from .NormLayer import NormLayer
from .FeedForward import FeedForward
from .MultiHeadAttention import MultiHeadAttention


class Decoder(torch.nn.Module):
    """
    A single Transformer decoder block.

    This module represents a Transformer decoder block consisting of:
    - Masked multi-head self-attention
    - Multi-head cross-attention with encoder output
    - Layer normalization with residual connections
    - Position-wise feedforward network

    Attributes:
        attention_masked (MultiHeadAttention): Masked multi-head self-attention mechanism.
        norm1 (NormLayer): Layer normalization after masked self-attention with residual connection.
        attention_cross (MultiHeadAttention): Cross-attention mechanism using encoder output.
        norm2 (NormLayer): Layer normalization after cross-attention with residual connection.
        ff (FeedForward): Position-wise feedforward network.
        norm3 (NormLayer): Layer normalization after feedforward network with residual connection.
    """

    def __init__(self, embed_dim: int, num_heads: int, d_k: int, d_v: int, dropout: float = 0.1):
        """Initializes the Decoder block.

        Args:
            embed_dim (int): Dimensionality of the input embeddings.
            num_heads (int): Number of attention heads.
            d_k (int): Dimensionality of key vectors per head.
            d_v (int): Dimensionality of value vectors per head.
            dropout (float, optional): Dropout rate applied to attention and feedforward layers. Defaults to 0.1.
        """
        super().__init__()
        self.attention_masked = MultiHeadAttention(embed_dim, num_heads, d_k, d_v, dropout=dropout, masked_attn=True)
        self.norm1 = NormLayer(embed_dim)

        self.attention_cross = MultiHeadAttention(embed_dim, num_heads, d_k, d_v, dropout=dropout, cross_attn=True)
        self.norm2 = NormLayer(embed_dim)

        self.ff = FeedForward(embed_dim, dropout=dropout)
        self.norm3 = NormLayer(embed_dim)

    def forward(self, dec_input: torch.Tensor, enc_output: torch.Tensor) -> torch.Tensor:
        """Applies the decoder block forward pass.

        Args:
            dec_input (torch.Tensor): Decoder input tensor of shape (batch_size, trg_seq_len, embed_dim).
            enc_output (torch.Tensor): Encoder output tensor of shape (batch_size, src_seq_len, embed_dim).

        Returns:
            torch.Tensor: Decoder output tensor of shape (batch_size, trg_seq_len, embed_dim).
        """
        # Masked self-attention + residual + norm
        attn_masked = self.attention_masked(dec_input)
        norm1_out = self.norm1(attn_masked + dec_input)

        # Cross-attention with encoder output + residual + norm
        attn_cross = self.attention_cross(norm1_out, enc_output)
        norm2_out = self.norm2(attn_cross + norm1_out)

        # Feedforward network + residual + norm
        ff_out = self.ff(norm2_out)
        dec_out = self.norm3(ff_out + norm2_out)

        return dec_out