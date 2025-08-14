"""
This module defines the Decoder class, which implements a single block of
the Transformer's decoder architecture.

The decoder block is a key component for sequence generation tasks. It
generates an output sequence from a contextualized input and is comprised
of the following key layers:
- Masked Multi-Head Self-Attention: This layer allows the decoder to attend
  to previous positions in the target sequence.
- Multi-Head Cross-Attention: This mechanism enables the decoder to
  incorporate information from the encoder's output.
- Position-wise Feedforward Network: A simple neural network that helps
  the model learn complex relationships.

Each of these layers is followed by dropout, residual connections, and layer
normalization, which are crucial for stable training in deep networks.
"""
import torch
from typing import Optional
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

    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 d_k: int,
                 d_v: int,
                 dropout: float = 0.1):
        """Initializes the Decoder block.

        Args:
            d_model (int): Dimensionality of the input embeddings.
            num_heads (int): Number of attention heads.
            d_k (int): Dimensionality of key vectors per head.
            d_v (int): Dimensionality of value vectors per head.
            dropout (float, optional): Dropout rate applied to attention and
                                        feedforward layers. Defaults to 0.1.
        """
        super().__init__()
        # Attention Layers
        self.attention_masked = MultiHeadAttention(d_model, num_heads, d_k, d_v,
                                                   dropout=dropout, masked_attn=True)
        self.attention_cross  = MultiHeadAttention(d_model, num_heads, d_k, d_v,
                                                  dropout=dropout, cross_attn=True)
        # FeedForward Layer
        self.ff = FeedForward(d_model, dropout=dropout)
        # Normalization Layers
        self.norm1            = torch.nn.LayerNorm(d_model)
        self.norm2            = torch.nn.LayerNorm(d_model)
        self.norm3            = torch.nn.LayerNorm(d_model)
        # Dropout
        self.dropout1         = torch.nn.Dropout(p=dropout)
        self.dropout2         = torch.nn.Dropout(p=dropout)
        self.dropout3         = torch.nn.Dropout(p=dropout)

    def forward(self,
                dec_input: torch.Tensor,
                enc_output: torch.Tensor,
                trg_padding_mask: Optional[torch.Tensor] = None,
                src_padding_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """Applies the decoder block forward pass.

        Args:
            dec_input (torch.Tensor): Decoder input tensor of shape
                                        (batch_size, trg_seq_len, d_model).
            enc_output (torch.Tensor): Encoder output tensor of shape
                                        (batch_size, src_seq_len, d_model).
            trg_padding_mask (Optional[torch.Tensor], optional): Padding mask for
                the decoder input. Shape: (batch_size, trg_seq_len). Default is None.
            src_padding_mask (Optional[torch.Tensor], optional): Padding mask for
                the encoder output. Shape: (batch_size, src_seq_len). Default is None.

        Returns:
            torch.Tensor: Decoder output tensor of shape
                            (batch_size, trg_seq_len, d_model).
        """
        # Masked self-attention + residual + norm
        attn_masked = self.attention_masked(dec_input, padding_mask=trg_padding_mask)
        norm1_out   = self.norm1(self.dropout1(attn_masked) + dec_input)

        # Cross-attention with encoder output + residual + norm
        attn_cross  = self.attention_cross(norm1_out, enc_output,
                                          padding_mask=src_padding_mask)
        norm2_out   = self.norm2(self.dropout2(attn_cross) + norm1_out)

        # Feedforward network + residual + norm
        ff_out      = self.ff(norm2_out)
        dec_out     = self.norm3(self.dropout3(ff_out) + norm2_out)

        return dec_out