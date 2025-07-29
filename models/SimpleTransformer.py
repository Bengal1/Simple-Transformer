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
from layers.PositionalEncoding import PositionalEncoding
from layers.Encoder import Encoder
from layers.Decoder import Decoder
from torch.onnx.symbolic_opset11 import unsqueeze


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
            embed_dim: int = 512,
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

        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.embed_scale = math.sqrt(embed_dim)

        # Token embeddings
        self.embedding_encoder = nn.Embedding(src_vocab_size, embed_dim,
                                              padding_idx=self.pad_token_id)
        self.embedding_decoder = nn.Embedding(trg_vocab_size, embed_dim,
                                              padding_idx=self.pad_token_id)

        # Positional encodings
        self.positional_encoding_encoder = PositionalEncoding(embed_dim)
        self.positional_encoding_decoder = PositionalEncoding(embed_dim)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Encoder and decoder stacks
        self.encoder_layers = nn.ModuleList([
            Encoder(embed_dim, num_heads, d_k, d_v, dropout)
            for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            Decoder(embed_dim, num_heads, d_k, d_v, dropout)
            for _ in range(num_layers)
        ])

        # Output projection - W_o
        self.w_o = nn.Linear(embed_dim, trg_vocab_size)

        # Xavier initialization - W_o
        init.xavier_uniform_(self.w_o.weight)

        # Normal initialization - Embedding
        init.normal_(self.embedding_encoder.weight, mean=0.0, std=0.02)
        self.embedding_encoder.weight.data *= math.sqrt(embed_dim)

        init.normal_(self.embedding_decoder.weight, mean=0.0, std=0.02)
        self.embedding_decoder.weight.data *= math.sqrt(embed_dim)

    @staticmethod
    def _generate_padding_mask(
                               seq: torch.Tensor,
                               pad_token_id: int) -> torch.Tensor:
        """
        Generates a boolean padding mask for a given sequence.
        True indicates padding (should be masked out).
        Shape: (batch_size, 1, 1, seq_len)
        """
        mask = (seq == pad_token_id).unsqueeze(1).unsqueeze(2) # (B, 1, 1, L)
        return mask

    def forward(self,
                src: torch.Tensor,
                trg: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Transformer model.

        Args:
            src (torch.Tensor): Source input tensor of shape (batch_size, src_seq_len).
            trg (torch.Tensor): Target input tensor of shape (batch_size, trg_seq_len).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, trg_seq_len, trg_vocab_size).
        """

        src_padding_mask = self._generate_padding_mask(src, self.pad_token_id)
        trg_padding_mask = self._generate_padding_mask(trg, self.pad_token_id)

        # Source embeddings + positional encoding
        src_embed        = self.embedding_encoder(src) * self.embed_scale
        src_pe           = self.positional_encoding_encoder(src_embed)
        src_pe_drop      = self.dropout1(src_pe)

        # Target embeddings + positional encoding
        trg_embed        = self.embedding_decoder(trg) * self.embed_scale
        trg_pe           = self.positional_encoding_decoder(trg_embed)
        trg_pe_drop      = self.dropout2(trg_pe)

        # Pass through stacked Encoders
        enc_output       = src_pe_drop
        for layer in self.encoder_layers:
            enc_output   = layer(enc_output, src_padding_mask)

        # Pass through stacked Decoders
        dec_output       = trg_pe_drop
        for layer in self.decoder_layers:
            dec_output   = layer(dec_output, enc_output,
                                 trg_padding_mask, src_padding_mask)

        # Output layer (no Softmax; handled by nn.CrossEntropyLoss)
        output           = self.w_o(dec_output)
        return output


    def translate(self,
                  src: torch.Tensor,
                  beam_size: int = 2,
                  max_len: int = None) -> torch.Tensor:
        """Translates source sequences using beam search decoding with length
            normalization.

        Args:
            src (torch.Tensor): Source tensor of shape (batch_size, src_seq_len).
            beam_size (int): Beam width for beam search.
            max_len (int): Maximum length of decoded sequences. If None, it's computed
                            dynamically.

        Returns:
            torch.Tensor: Tensor of shape (batch_size, decoded_seq_len) with
                        predicted token IDs.
        """
        if max_len is None or max_len <= 0:
            max_len = int(src.size(1) * 1.6) + 10

        with torch.no_grad():
            batch_size = src.size(0)
            vocab_size = self.w_o.out_features
            device = src.device

            # === Encode the input ===
            src_padding_mask = self._generate_padding_mask(src, self.pad_token_id)
            src_embed = self.dropout1(self.positional_encoding_encoder(
                        self.embedding_encoder(src) * self.embed_scale))
            enc_output = src_embed
            for layer in self.encoder_layers:
                enc_output = layer(enc_output, src_padding_mask)

            # Repeat encoder output for each beam
            enc_output = enc_output.unsqueeze(1).repeat(1, beam_size, 1, 1)  # (B, beam, L, D)
            enc_output = enc_output.view(batch_size * beam_size,
                                         *enc_output.shape[2:])  # (B*beam, L, D)

            src_padding_mask_beams = src_padding_mask.repeat(1, beam_size, 1, 1)  # (B, beam, 1, L_src)
            src_padding_mask_beams = src_padding_mask_beams.view(
                batch_size * beam_size,
                *src_padding_mask_beams.shape[2:]).unsqueeze(1)  # (B*beam, 1, 1, L_src)

            # === Initialize decoder inputs ===
            sequences = torch.full((batch_size * beam_size, 1), self.bos_token_id,
                                   dtype=torch.long, device=device)
            trg_padding_mask_beams = (sequences == self.pad_token_id).unsqueeze(
                1).unsqueeze(2)

            sequence_scores = torch.zeros(batch_size, beam_size, device=device)
            sequence_scores[:, 1:] = float('-inf')  # only keep 1st beam
            sequence_scores = sequence_scores.view(-1)  # (B*beam,)
            finished = torch.zeros_like(sequence_scores, dtype=torch.bool)

            alpha = 0.6  # Length penalty factor

            for _ in range(max_len):
                # Decoder embedding
                trg_embed = self.dropout2(self.positional_encoding_decoder(
                            self.embedding_decoder(sequences) * self.embed_scale))
                dec_output = trg_embed
                for layer in self.decoder_layers:
                    dec_output = layer(dec_output, enc_output,
                                       trg_padding_mask_beams, src_padding_mask_beams)

                logits = self.w_o(dec_output[:, -1, :])  # Only last token logits
                log_probs = F.log_softmax(logits, dim=-1)

                # Prevent expansion of finished beams
                log_probs[finished] = float('-inf')
                log_probs[finished, self.eos_token_id] = 0  # allow only <eos>

                # Expand beams
                scores = sequence_scores.unsqueeze(1) + log_probs  # (B*beam, V)
                scores = scores.view(batch_size, -1)  # (B, beam * V)

                top_scores, top_indices = scores.topk(beam_size, dim=-1)  # select top beams
                beam_indices = top_indices // vocab_size
                token_indices = top_indices % vocab_size

                # Gather previous sequences
                batch_offset = torch.arange(batch_size, device=device
                                            ).unsqueeze(1) * beam_size
                gather_indices = (beam_indices + batch_offset).view(-1)
                next_tokens = token_indices.view(-1)

                sequences = sequences[gather_indices]
                sequences = torch.cat([sequences,
                                       next_tokens.unsqueeze(1)], dim=-1)

                trg_padding_mask_beams = (sequences == self.pad_token_id).unsqueeze(
                    1).unsqueeze(2)

                finished = finished[gather_indices]
                sequence_scores = top_scores.view(-1)

                # Update finished
                finished |= (next_tokens == self.eos_token_id)

                # === Length penalty ===
                lengths = sequences.new_full((batch_size * beam_size,)
                                             , sequences.size(1), dtype=torch.float)
                eos_mask = sequences == self.eos_token_id
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

            return torch.nn.utils.rnn.pad_sequence(best_sequences, batch_first=True,
                                                   padding_value=self.pad_token_id)


    # --- Model Utilities ---
    def count_parameters(self, only_trainable: bool = True) -> int:
        """
        Counts the total number of learnable parameters in the model.

        Args:
            only_trainable (bool): If True, only counts parameters that
                                   require gradients (i.e., are trainable).

        Returns:
            int: The total number of parameters.
        """
        if only_trainable:
            num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            num_params = sum(p.numel() for p in self.parameters())
        print(f"Number of trainable parameters: {num_params:,}")
        return num_params