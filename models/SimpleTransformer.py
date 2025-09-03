"""
This module defines the SimpleTransformer class, an implementation of the
Transformer model architecture based on the paper "Attention Is All You Need".

The model is a sequence-to-sequence architecture with a stacked encoder
and decoder, and includes the following key components:
- Encoder-Decoder Architecture: A standard sequence-to-sequence architecture
  with a stacked encoder and decoder.
- Embeddings: Dedicated embedding layers for both the source and target
  vocabularies.
- Positional Encoding: Positional encoding modules are incorporated
  to capture sequence order information.

The model is designed to handle the full training and inference workflow,
and provides two primary functions for this purpose:
- `forward`: A training method that returns raw logits without applying a
  softmax, designed to work with `torch.nn.CrossEntropyLoss`.
- `translate`: An inference method that generates translations using a
  length-normalized beam search.
"""
import math
import torch
import torch.nn.functional as F
from typing import Optional
from .layers.Encoder import Encoder
from .layers.Decoder import Decoder
from .layers.PositionalEncoding import PositionalEncoding
# from .layers import Encoder, Decoder, MultiHeadAttention, FeedForward

class SimpleTransformer(torch.nn.Module):
    """A simplified Transformer model for sequence-to-sequence tasks.

    This model consists of a standard Transformer architecture with stacked
    encoder and decoder layers, token embeddings, and positional encodings,
    designed for tasks such as machine translation.

    Attributes:
        pad_token_id (int): The ID for the padding token (default: 0).
        bos_token_id (int): The ID for the beginning-of-sequence token (default: 1).
        eos_token_id (int): The ID for the end-of-sequence token (default: 2).
        unk_token_id (int): The ID for the unknown token (default: 3).
        embed_scale (float): A scaling factor for the embeddings,
                             calculated as sqrt(embed_dim).

        embedding_encoder (nn.Embedding): Embedding layer for source tokens.
        embedding_decoder (nn.Embedding): Embedding layer for target tokens.
        positional_encoding_encoder (PositionalEncoding): Positional encoding
                                                         module for the encoder.
        positional_encoding_decoder (PositionalEncoding): Positional encoding
                                                         module for the decoder.
        dropout_enc (nn.Dropout): Dropout layer applied in the encoder.
        dropout_dec (nn.Dropout): Dropout layer applied in the decoder.
        encoder_layers (nn.ModuleList[Encoder]): Stacked list of encoder layers.
        decoder_layers (nn.ModuleList[Decoder]): Stacked list of decoder layers.
        w_o (nn.Linear): Final linear layer mapping decoder output to the target
                         vocabulary size.
                         Note: This layer produces raw logits;a softmax is not applied.
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
            d_ff: int =2048,
            dropout: float = 0.1):
        """Initializes the SimpleTransformer model's layers and parameters.

        This constructor sets up all the sub-modules required for the Transformer,
        including embeddings, positional encodings, dropout layers,
        encoder/decoder stacks, and the final projection layer. It also
        performs weight initialization.

        Args:
            src_vocab_size (int): The size of the source vocabulary.
            trg_vocab_size (int): The size of the target vocabulary.
            embed_dim (int): The dimensionality of token embeddings and model states.
            num_heads (int): The number of attention heads to use.
            num_layers (int): The number of identical encoder and decoder layers to stack.
            d_k (int): The dimensionality of key vectors in the attention mechanism.
            d_v (int): The dimensionality of value vectors in the attention mechanism.
            d_ff (int): The hidden layer dimension of the feed-forward networks.
            dropout (float): The dropout rate to apply to both the encoder and
                             decoder inputs.
        """
        super().__init__()

        # Define special token IDs and embedding scale
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3
        self.embed_scale = math.sqrt(embed_dim)

        # Token embeddings for source and target vocabularies
        self.embedding_encoder = torch.nn.Embedding(
            src_vocab_size, embed_dim, padding_idx=self.pad_token_id)
        self.embedding_decoder = torch.nn.Embedding(
            trg_vocab_size, embed_dim, padding_idx=self.pad_token_id)

        # Positional encoding modules for adding sequence position information
        self.positional_encoding_encoder = PositionalEncoding(embed_dim)
        self.positional_encoding_decoder = PositionalEncoding(embed_dim)

        # Dropout layers for encoder and decoder inputs
        self.dropout_enc = torch.nn.Dropout(dropout)
        self.dropout_dec = torch.nn.Dropout(dropout)

        # Stacked encoder and decoder layers
        self.encoder_layers = torch.nn.ModuleList([
            Encoder(embed_dim, num_heads, d_k, d_v, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.decoder_layers = torch.nn.ModuleList([
            Decoder(embed_dim, num_heads, d_k, d_v, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Final projection layer to map model output to vocabulary logits.
        self.w_o = torch.nn.Linear(embed_dim, trg_vocab_size)

        # --- Weight Initialization ---
        # Initialize final projection layer with Xavier uniform initialization
        torch.nn.init.xavier_uniform_(self.w_o.weight)

        # Initialize embedding layers with a scaled normal distribution
        torch.nn.init.normal_(self.embedding_encoder.weight, mean=0.0, std=0.02)
        self.embedding_encoder.weight.data *= self.embed_scale

        torch.nn.init.normal_(self.embedding_decoder.weight, mean=0.0, std=0.02)
        self.embedding_decoder.weight.data *= self.embed_scale


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
        # Encode the source sequence to get its contextual representation.
        encoder_output, src_padding_mask = self._encode_source(src)

        # Decode the target sequence using the encoder output for cross-attention.
        decoder_output = self._decode_step(trg, encoder_output, src_padding_mask)

        # Project the decoder output to the vocabulary size to get logits.
        # Note: Softmax is not applied here; it's handled by nn.CrossEntropyLoss.
        output = self.w_o(decoder_output)

        return output


    def translate(self,
                  src: torch.Tensor,
                  beam_size: int = 4,
                  max_len: int = None) -> torch.Tensor:
        """Performs length-normalized beam search decoding to translate source sequences.

        Args:
            src (torch.Tensor): Source tensor of shape (batch_size, src_seq_len).
            beam_size (int): The number of beams to maintain during decoding.
            max_len (int, optional): The maximum length for the decoded sequences.
                                     If None, a dynamic length is calculated.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, decoded_seq_len) with
                          predicted token IDs, padded to the maximum length.
        """
        # Dynamically calculate max_len if not provided.
        if max_len is None or max_len <= 0:
            max_len = int(src.size(1) * 1.6) + 10

        with torch.no_grad():
            batch_size = src.size(0)
            device = src.device

            # --- Encoding Phase ---
            # Generate source padding mask and compute encoder output.
            enc_output, src_padding_mask = self._encode_source(src)

            # Expand encoder output and padding mask for all beams.
            enc_output = self._expand_to_beams(enc_output, beam_size)
            src_padding_mask_beams = self._expand_to_beams(src_padding_mask, beam_size)

            # --- Decoding Initialization ---
            # Initialize sequences for all beams with the BOS token.
            sequences = torch.full((batch_size * beam_size, 1), self.bos_token_id,
                                   dtype=torch.long, device=device)

            # Initialize beam scores (only the first beam active) and finished status.
            sequence_scores = torch.zeros(batch_size, beam_size, device=device)
            sequence_scores[:, 1:] = float('-inf')  # Only the first beam is active.
            sequence_scores = sequence_scores.view(-1)
            finished = torch.zeros_like(sequence_scores, dtype=torch.bool)

            # --- Main Decoding Loop ---
            for _ in range(max_len):
                # Pass sequences through the decoder to predict the next token.
                dec_output = self._decode_step(sequences, enc_output,
                                               src_padding_mask_beams)

                # Get log probabilities for the next token.
                logits = self.w_o(dec_output[:, -1, :])
                log_probs = F.log_softmax(logits, dim=-1)

                (sequences, sequence_scores, finished, next_tokens) = self._beam_search_step(
                    sequences,
                    sequence_scores,
                    log_probs,
                    finished,
                    beam_size
                )
                finished |= (next_tokens == self.eos_token_id)

                # Normalize scores with length penalty.
                length_penalty = self._calculate_length_penalty(sequences)
                normalized_scores = sequence_scores / length_penalty

                # Exit early if all beams have finished.
                if finished.view(batch_size, beam_size).all(dim=1).all():
                    break

            # --- Final Selection ---
            # Choose the best sequence for each batch based on normalized scores.
            normalized_scores = normalized_scores.view(batch_size, beam_size)
            best_beam = normalized_scores.argmax(dim=1)
            best_sequences = [sequences[i * beam_size + best_beam[i]] for i in
                              range(batch_size)]

            return torch.nn.utils.rnn.pad_sequence(best_sequences, batch_first=True,
                                                   padding_value=self.pad_token_id)


    # --- Private Helper Methods ---
    @staticmethod
    def _generate_padding_mask(
            seq: torch.Tensor,
            pad_token_id: int) -> torch.Tensor:
        """Generates a boolean padding mask for a given sequence.

        This function creates a boolean tensor that indicates which elements
        of the sequence are padding tokens. The mask is then reshaped with
        singleton dimensions to be compatible with a multi-head attention
        mechanism.

        Args:
            seq (torch.Tensor): The input sequence tensor of shape
                                (batch_size, seq_len).
            pad_token_id (int): The ID of the padding token.

        Returns:
            torch.Tensor: A boolean mask tensor of shape
                          (batch_size, 1, 1, seq_len), where `True` indicates
                          a padding token.
        """
        # Create a boolean mask where True indicates padding tokens.
        # The mask is then reshaped to (B, 1, 1, L) for attention broadcasting.
        mask = (seq == pad_token_id).unsqueeze(1).unsqueeze(2)
        return mask


    def _encode_source(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encodes the source sequence using the transformer encoder.

        This function applies the embedding, positional encoding, and then passes the
        input through all stacked encoder layers to produce the final encoded representation.

        Args:
            src (torch.Tensor): The source sequence tensor of shape (batch_size, src_seq_len).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - enc_output (torch.Tensor): The final encoded output from the
                  encoder stack, with shape (batch_size, src_seq_len, embedding_dim).
                - src_padding_mask (torch.Tensor): The padding mask for the source
                  sequence, with shape (batch_size, 1, 1, src_seq_len).
        """
        src_padding_mask = self._generate_padding_mask(src, self.pad_token_id)

        # Apply embedding, positional encoding, and dropout to the source sequence
        enc_output = self.dropout_enc(
            self.positional_encoding_encoder(
                self.embedding_encoder(src) * self.embed_scale))

        # Pass the output through all encoder layers
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_padding_mask)

        return enc_output, src_padding_mask


    def _decode_step(self,
                     targets: torch.Tensor,
                     enc_output: torch.Tensor,
                     src_padding_mask: torch.Tensor,
                     trg_padding_mask: Optional[torch.Tensor] = None
                     ) -> torch.Tensor:
        """Performs a single forward pass through the decoder.

        This function processes the target sequence up to the current timestep,
        applies self-attention and cross-attention, and returns the final raw
        decoder output.

        Args:
            targets (torch.Tensor): The target sequence tensor of shape
                                    (batch_size, trg_seq_len).
            enc_output (torch.Tensor): The encoded source output from the
                                       transformer encoder, with shape
                                       (batch_size, src_seq_len, embedding_dim).
            src_padding_mask (torch.Tensor): The padding mask for the source
                                             sequence, with shape
                                             (batch_size, 1, 1, src_seq_len).
            trg_padding_mask (Optional[torch.Tensor]): The padding mask for
                                                       the target sequence.
                                                       If None, it is generated
                                                       automatically.

        Returns:
            torch.Tensor: The final decoder output, with shape
                          (batch_size, trg_seq_len, embedding_dim).
        """
        if trg_padding_mask is None:
            trg_padding_mask = self._generate_padding_mask(targets,
                                                           self.pad_token_id)

        # Apply embedding, positional encoding, and dropout to the target sequence
        dec_output = self.dropout_dec(
            self.positional_encoding_decoder(
                self.embedding_decoder(targets) * self.embed_scale))

        # Pass through all decoder layers
        for layer in self.decoder_layers:
            dec_output = layer(
                dec_output, enc_output, trg_padding_mask, src_padding_mask)

        return dec_output

    @staticmethod
    def _expand_to_beams(tensor: torch.Tensor,
                          beam_size: int) -> torch.Tensor:
        """Expands a tensor to match the beam search dimension.

        This function takes a tensor of shape (batch_size, ...) and expands it
        to (batch_size * beam_size, ...) by repeating each item in the batch
        `beam_size` times. This is typically used to prepare inputs for the
        first decoding step of a beam search.

        Args:
            tensor (torch.Tensor): The input tensor to expand, with shape
                                   (batch_size, ...).
            beam_size (int): The number of beams to expand the tensor for.

        Returns:
            torch.Tensor: The expanded and reshaped tensor, with shape
                          (batch_size * beam_size, ...).
        """
        # Add a beam dimension, reshaping from (B, ...) to (B, 1, ...).
        expanded_tensor = tensor.unsqueeze(1)

        # Repeat the tensor `beam_size` times along the new beam dimension.
        expanded_tensor = expanded_tensor.repeat(1, beam_size,
                                                 *[1] * (tensor.ndim - 1))

        # Flatten the batch and beam dimensions into one.
        reshaped_tensor = expanded_tensor.view(-1, *expanded_tensor.shape[2:])

        return reshaped_tensor


    def _beam_search_step(self,
                          sequences: torch.Tensor,
                          sequence_scores: torch.Tensor,
                          log_probs: torch.Tensor,
                          finished: torch.Tensor,
                          beam_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Updates the beams based on the next token predictions.

        This function combines previous beam scores with the log probabilities
        of the next predicted tokens, selects the `beam_size` highest-scoring
        new beams for each batch item, and updates the sequence states.

        Args:
            sequences (torch.Tensor): Current decoded sequences of shape
                                      (batch_size * beam_size, sequence_length).
            sequence_scores (torch.Tensor): Scores of the current sequences, of
                                            shape (batch_size * beam_size).
            log_probs (torch.Tensor): Log probabilities of the next possible tokens,
                                      of shape (batch_size * beam_size, vocab_size).
            finished (torch.Tensor): A boolean tensor indicating which sequences
                                     in the beams have finished.
            beam_size (int): The number of beams to maintain.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - new_sequences (torch.Tensor): The updated sequences with the
                  next tokens appended.
                - new_sequence_scores (torch.Tensor): The scores of the new beams.
                - new_finished (torch.Tensor): The updated finished status for
                  the new beams.
                - next_tokens (torch.Tensor): The token IDs of the selected next tokens.
        """
        total_beams = log_probs.size(0)
        vocab_size = log_probs.size(-1)
        batch_size = total_beams // beam_size

        # Mask finished beams and allow the EOS token.
        log_probs[finished] = float('-inf')
        log_probs[finished, self.eos_token_id] = 0

        # Combine previous sequence scores with new token log probabilities.
        scores = sequence_scores.unsqueeze(1) + log_probs
        scores = scores.view(batch_size, -1)

        # Find the top `beam_size` scores for each batch item.
        top_scores, top_indices = scores.topk(beam_size, dim=-1)

        # Determine parent beams and next tokens.
        parent_beam_indices = top_indices // vocab_size
        next_token_indices = top_indices % vocab_size

        # Calculate indices to gather previous states.
        batch_offset = (
                torch.arange(batch_size, device=sequences.device)
                .unsqueeze(1) * beam_size
        )
        gather_indices = (parent_beam_indices + batch_offset).view(-1)

        # Get and flatten next token indices.
        next_tokens = next_token_indices.view(-1)

        # Gather previous states to update sequences, scores, and finished status.
        new_sequences = sequences[gather_indices]
        new_sequences = torch.cat(
            [new_sequences, next_tokens.unsqueeze(1)],
            dim=-1
        )
        new_finished = finished[gather_indices]
        new_sequence_scores = top_scores.view(-1)

        return new_sequences, new_sequence_scores, new_finished, next_tokens


    def _calculate_length_penalty(self,
                                  sequences: torch.Tensor,
                                  alpha: float = 0.6) -> torch.Tensor:
        """Calculates a length penalty factor for each sequence.

        This function computes the effective length of each sequence, which is the
        position of the first end-of-sequence (EOS) token, or the full sequence
        length if no EOS token is present. It then applies a length normalization
        formula to penalize shorter sequences during beam search.

        Args:
            sequences (torch.Tensor): A tensor of shape (batch_size * beam_size,
                                      sequence_length) containing token IDs.
            alpha (float, optional): The length penalty exponent. A higher value
                                     results in a stronger penalty for shorter
                                     sequences. Defaults to 0.6.

        Returns:
            torch.Tensor: A tensor of shape (batch_size * beam_size,) containing
                          the length penalty factor for each sequence.
        """
        # Initialize a tensor with the current sequence length for all beams
        current_lengths = sequences.new_full(
            (sequences.size(0),), sequences.size(1), dtype=torch.float)

        # Create a mask to identify sequences that have reached the EOS token
        eos_mask = sequences == self.eos_token_id
        has_eos = eos_mask.any(dim=1)

        # Find the first EOS token position for sequences that have one
        eos_positions = eos_mask.float().argmax(dim=1)

        # Update the length for these sequences to be (position + 1)
        current_lengths[has_eos] = (eos_positions[has_eos] + 1).float()

        # Apply the length normalization formula
        length_penalty_factors = ((5.0 + current_lengths) / 6.0).pow(alpha)

        return length_penalty_factors


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