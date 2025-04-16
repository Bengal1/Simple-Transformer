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


class PositionalEncoding(torch.nn.Module):
    """
    Adds positional encoding to the input tensor.

    The positional encoding follows the formula from "Attention Is All You Need"
    and helps the Transformer model retain positional information.

    Attributes:
        embed_dim (int): The embedding dimension of the model.
        n (int): The base for the sinusoidal encoding.
    """

    def __init__(self, embed_dim: int, n: int = 10000):
        """Initializes the positional encoding module.

        Args:
            embed_dim (int): The embedding dimension of the model.
            n (int, optional): The base for the sinusoidal encoding. Default is 10000.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.n = n

    def _create_positional_encoding(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Creates the positional encoding matrix.

        The encoding is based on sinusoidal functions that encode relative
        position information for each token.

        Args:
            seq_len (int): Length of the sequence for positional encoding.
            device (torch.device): Device where the tensor should be allocated.

        Returns:
            torch.Tensor: The positional encoding matrix of shape (seq_len, embed_dim).
        """
        pos_encoding = torch.zeros(seq_len, self.embed_dim, device=device)
        k_pos = torch.arange(seq_len, device=device).unsqueeze(dim=1).float()
        _2i = torch.arange(0, self.embed_dim, step=2, device=device).float()

        pos_encoding[:, 0::2] = torch.sin(k_pos / self.n ** (_2i / self.embed_dim))
        pos_encoding[:, 1::2] = torch.cos(k_pos / self.n ** (_2i / self.embed_dim))

        return pos_encoding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).

        Returns:
            torch.Tensor: Tensor with added positional encoding,
            of shape (batch_size, seq_len, embed_dim).
        """
        batch_size, seq_len, _ = x.shape
        pos_encoding = self._create_positional_encoding(seq_len, x.device)

        return x + pos_encoding.unsqueeze(0)  # Broadcast across batch


class FeedForward(torch.nn.Module):
    """
    A FeedForward neural network used in the Transformer model.

    This network consists of two fully connected layers with a ReLU activation
    and dropout in between.

    Attributes:
        fc1 (nn.Linear): First fully connected layer that expands the input dimension.
        fc2 (nn.Linear): Second fully connected layer that projects back to the original dimension.
        relu (nn.ReLU): ReLU activation function.
        dropout (nn.Dropout): Dropout layer for regularization.
    """

    def __init__(self, d_model: int, hidden_dim: int = 2048, dropout: float = 0.1):
        """Initializes the FeedForward network.

        Args:
            d_model (int): The input and output feature dimension.
            hidden_dim (int, optional): The hidden layer dimension. Default is 512.
            dropout (float, optional): The dropout probability. Default is 0.1.
        """
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the FeedForward network.

        The input tensor is passed through a linear layer, followed by ReLU activation,
        dropout, and a final linear layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class NormLayer(torch.nn.Module):
    """
    Implements layer normalization used in the Transformer.

    This normalization technique stabilizes the training process by normalizing
    inputs across the last dimension and scaling them with learnable parameters.

    Attributes:
        gamma (nn.Parameter): Learnable scale parameter initialized to ones.
        beta (nn.Parameter): Learnable shift parameter initialized to zeros.
        epsilon (float): A small value added to variance for numerical stability.
    """

    def __init__(self, d_model: int, epsilon: float = 1e-15):
        """Initializes the layer normalization module.

        Args:
            d_model (int): The dimension of the input tensor.
            epsilon (float, optional): A small value added to variance for numerical stability. Default is 1e-15.
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies layer normalization to the input tensor.

        Normalizes the input across the last dimension and applies learnable
        scaling (`gamma`) and shifting (`beta`).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Normalized tensor of the same shape as the input.
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        normalized = (x - mean) / torch.sqrt(var + self.epsilon)
        return self.gamma * normalized + self.beta


class MultiHeadAttention(torch.nn.Module):
    """
    Multi-Head Attention module.

    This module implements the multi-head attention mechanism used in Transformer models.
    It supports both self-attention and cross-attention.

    Attributes:
        num_heads (int): Number of attention heads.
        d_k (int): Dimension of key vectors per head.
        d_v (int): Dimension of value vectors per head.
        cross_attn (bool): If True, performs cross-attention (decoder).
        masked_attn (bool): If True, applies a causal mask for autoregressive decoding.
        w_q (nn.ModuleList): Linear layers for projecting queries.
        w_k (nn.ModuleList): Linear layers for projecting keys.
        w_v (nn.ModuleList): Linear layers for projecting values.
        w_out (nn.Linear): Linear layer for final projection after attention computation.
    """

    def __init__(self, embed_dim: int, num_heads: int = 1, d_k: int = 64,
                 d_v: int = 128, dropout: float = 0.0, cross_attn: bool = False, masked_attn: bool = False):
        """Initializes the MultiHeadAttention module.

        Args:
            embed_dim (int): Dimension of input embeddings.
            num_heads (int, optional): Number of attention heads. Default is 1.
            d_k (int, optional): Dimension of key vectors per head. Default is 64.
            d_v (int, optional): Dimension of value vectors per head. Default is 128.
            cross_attn (bool, optional): If True, enables cross-attention mode (decoder). Default is False.
            masked_attn (bool, optional): If True, applies a causal mask for autoregressive decoding. Default is False.
        """
        super().__init__()

        self.num_heads = num_heads
        self.d_k = d_k  # Size of key vectors per head
        self.d_v = d_v  # Size of value vectors per head
        self.cross_attn = cross_attn
        self.masked_attn = masked_attn

        # Linear projections for Q, K, V for each head
        self.w_q = nn.ModuleList([nn.Linear(embed_dim, d_k) for _ in range(num_heads)])
        self.w_k = nn.ModuleList([nn.Linear(embed_dim, d_k) for _ in range(num_heads)])
        self.w_v = nn.ModuleList([nn.Linear(embed_dim, d_v) for _ in range(num_heads)])

        self.w_out = nn.Linear(d_v * num_heads, embed_dim)
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def _scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                      mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes scaled dot-product attention.

        Args:
            Q (torch.Tensor): Query tensor of shape (batch_size, num_heads, src_len, d_k).
            K (torch.Tensor): Key tensor of shape (batch_size, num_heads, tgt_len, d_k).
            V (torch.Tensor): Value tensor of shape (batch_size, num_heads, tgt_len, d_v).
            mask (torch.Tensor, optional): Attention mask of shape (1, 1, src_len, tgt_len), with -inf for masked positions.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Attention output of shape (batch_size, num_heads, src_len, d_v).
                - Attention probabilities of shape (batch_size, num_heads, src_len, tgt_len).
        """
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=Q.dtype, device=Q.device))

        if mask is not None:
            attn_scores = attn_scores + mask

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs) # Dropout
        return torch.matmul(attn_probs, V), attn_probs

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """Performs forward pass of multi-head attention.

        Args:
            x (torch.Tensor): Source tensor of shape (batch_size, src_len, embed_dim).
            y (torch.Tensor, optional): Target tensor for cross-attention, shape (batch_size, tgt_len, embed_dim).
                                        If None, self-attention is performed.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, src_len, embed_dim).
        """
        batch_size, src_len, _ = x.shape
        _, trg_len, _ = y.shape if self.cross_attn else x.shape

        # Initialize Q, K, V for each head
        Q, K, V = [], [], []

        for i in range(self.num_heads):
            Q.append(self.w_q[i](x))
            K.append(self.w_k[i](x if not self.cross_attn else y))
            V.append(self.w_v[i](x if not self.cross_attn else y))

        # Stack the Q, K, V tensors into one tensor of shape (batch_size, num_heads, max_length, d_k/d_v)
        Q = torch.stack(Q, dim=1)
        K = torch.stack(K, dim=1)
        V = torch.stack(V, dim=1)

        # Create dynamic mask of shape (1, 1, src_len, trg_len)
        mask = None
        if self.masked_attn:
            mask = torch.triu(torch.full((src_len, trg_len), float('-inf'), device=x.device), diagonal=1)
            mask = mask.unsqueeze(0).unsqueeze(1)  # (1, 1, src_len, trg_len)

        # Apply scaled dot-product attention
        attention_output, _ = self._scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads and project to output dimension + Dropout
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, src_len, -1)
        output = self.w_out(attention_output)
        return self.out_dropout(output) # Dropout


class Encoder(nn.Module):
    """
    A single Transformer encoder block consisting of:
    - Multi-head self-attention
    - Layer normalization and residual connections
    - Position-wise feedforward network
    """

    def __init__(self, embed_dim: int, num_heads: int, d_k: int, d_v: int, dropout: float = 0.0):
        """
        Initializes the Encoder block.

        Args:
            embed_dim (int): Dimensionality of input embeddings.
            num_heads (int): Number of attention heads.
            d_k (int): Dimensionality of key vectors.
            d_v (int): Dimensionality of value vectors.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, d_k, d_v, dropout)
        self.norm1 = NormLayer(embed_dim)

        self.ff = FeedForward(embed_dim, dropout=dropout)
        self.norm2 = NormLayer(embed_dim)

    def forward(self, enc_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder block.

        Args:
            enc_input (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).

        Returns:
            torch.Tensor: Output tensor with the same shape as input.
        """
        # Multi-head self-attention + residual + norm
        attn_out = self.attention(enc_input)
        norm1_out = self.norm1(attn_out + enc_input)

        # Feedforward network + residual + norm
        ff_out = self.ff(norm1_out)
        enc_out = self.norm2(ff_out + norm1_out)

        return enc_out


class Decoder(nn.Module):
    """
    A single Transformer decoder block consisting of:
    - Masked multi-head self-attention
    - Multi-head cross-attention with encoder output
    - Layer normalization and residual connections
    - Position-wise feedforward network
    """

    def __init__(self, embed_dim: int, num_heads: int, d_k: int, d_v: int, dropout: float = 0.1):
        """
        Initializes the Decoder block.

        Args:
            embed_dim (int): Dimensionality of input embeddings.
            num_heads (int): Number of attention heads.
            d_k (int): Dimensionality of key vectors.
            d_v (int): Dimensionality of value vectors.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.attention_masked = MultiHeadAttention(embed_dim, num_heads, d_k, d_v, dropout, masked_attn=True)
        self.norm1 = NormLayer(embed_dim)

        self.attention_cross = MultiHeadAttention(embed_dim, num_heads, d_k, d_v, dropout, cross_attn=True)
        self.norm2 = NormLayer(embed_dim)

        self.ff = FeedForward(embed_dim, dropout=dropout)
        self.norm3 = NormLayer(embed_dim)

    def forward(self, dec_input: torch.Tensor, enc_output: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder block.

        Args:
            dec_input (torch.Tensor): Decoder input tensor of shape (batch_size, trg_seq_len, embed_dim).
            enc_output (torch.Tensor): Encoder output tensor of shape (batch_size, src_seq_len, embed_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, trg_seq_len, embed_dim).
        """
        # Masked self-attention + residual + norm
        attn_masked = self.attention_masked(dec_input)
        norm1 = self.norm1(attn_masked + dec_input)

        # Cross-attention with encoder output + residual + norm
        attn_cross = self.attention_cross(norm1, enc_output)
        norm2 = self.norm2(attn_cross + norm1)

        # Feedforward network + residual + norm
        ff_out = self.ff(norm2)
        dec_out = self.norm3(ff_out + norm2)

        return dec_out


class SimpleTransformer(nn.Module):
    """
    A simplified Transformer model for sequence-to-sequence tasks like translation.
    It consists of stacked encoder and decoder layers, embeddings, and a final linear projection.

    Attributes:
        embedding_encoder (nn.Embedding): Embedding layer for source tokens.
        positional_encoding_encoder (PositionalEncoding): Adds position information to source embeddings.
        embedding_decoder (nn.Embedding): Embedding layer for target tokens.
        positional_encoding_decoder (PositionalEncoding): Adds position information to target embeddings.
        encoder_layers (nn.ModuleList): Stacked encoder blocks.
        decoder_layers (nn.ModuleList): Stacked decoder blocks.
        w_o (nn.Linear): Final linear layer projecting decoder output to vocabulary logits.
    """

    def __init__(self, src_vocab_size: int, trg_vocab_size: int, embed_dim: int,
                 num_heads: int = 8, num_layers: int = 6, d_k: int = 32, d_v: int = 64,
                 dropout: float = 0.1
    ):
        """
        Initializes the SimpleTransformer model.

        Args:
            src_vocab_size (int): Size of the source vocabulary.
            trg_vocab_size (int): Size of the target vocabulary.
            embed_dim (int): Dimensionality of token embeddings.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of encoder/decoder layers to stack.
            d_k (int): Dimensionality of keys.
            d_v (int): Dimensionality of values.
            dropout (float): Dropout probability for regularization.
        """

        super().__init__()

        # Embedding layers for source and target
        self.embedding_encoder = nn.Embedding(src_vocab_size, embed_dim, padding_idx=1)
        self.embedding_decoder = nn.Embedding(trg_vocab_size, embed_dim, padding_idx=1)

        # Positional encoding
        self.positional_encoding_encoder = PositionalEncoding(embed_dim)
        self.positional_encoding_decoder = PositionalEncoding(embed_dim)

        # Dropout after embedding + positional encoding
        self.dropout = nn.Dropout(dropout)

        # Stacked encoder layers
        self.encoder_layers = nn.ModuleList([
            Encoder(embed_dim, num_heads, d_k, d_v) for _ in range(num_layers)
        ])

        # Stacked decoder layers
        self.decoder_layers = nn.ModuleList([
            Decoder(embed_dim, num_heads, d_k, d_v) for _ in range(num_layers)
        ])

        # Output block
        self.w_o = nn.Linear(embed_dim, trg_vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer model.

        Args:
            src (torch.Tensor): Source input tensor of shape (batch_size, src_seq_len).
            trg (torch.Tensor): Target input tensor of shape (batch_size, trg_seq_len).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, trg_seq_len, trg_vocab_size).
        """
        # Source embeddings + positional encoding
        src_embed = self.embedding_encoder(src)
        src_pe = self.positional_encoding_encoder(src_embed)
        # src_pe = self.dropout(src_pe)

        # Target embeddings + positional encoding
        trg_embed = self.embedding_decoder(trg)
        trg_pe = self.positional_encoding_decoder(trg_embed)
        # trg_pe = self.dropout(trg_pe)

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

    def translate(self, src: torch.Tensor) -> torch.Tensor:
        """Translates a given source sequence into the target language using greedy decoding.

        Args:
            src (torch.Tensor): Source sequence tensor of shape (batch_size, src_seq_len).

        Returns:
            torch.Tensor: Predicted target sequence of shape (batch_size, trg_seq_len).
        """
        with torch.no_grad():
            unk_token_id, bos_token_id, eos_token_id = 0, 2, 3  # <bos> and <eos> token IDs
            batch_size, max_target_length = src.shape  # Extract input dimensions

            # Encoder step
            src_embed = self.embedding_encoder(src)
            src_pe = self.positional_encoding_encoder(src_embed)
            enc_output = src_pe
            for layer in self.encoder_layers:
                enc_output = layer(enc_output)

            # Decoder initialization
            trg_seq = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=src.device)

            # Track whether <eos> is generated in any sequence
            generated_eos = torch.zeros(batch_size, dtype=torch.bool, device=src.device)

            for _ in range(max_target_length):
                # Decoder step
                trg_embed = self.embedding_decoder(trg_seq)
                trg_pe = self.positional_encoding_decoder(trg_embed)

                dec_output = trg_pe
                for layer in self.decoder_layers:
                    dec_output = layer(dec_output, enc_output)

                # Predict next token
                out_logits = self.w_o(dec_output[:, -1, :])
                probs = self.softmax(out_logits)
                next_token = probs.argmax(dim=-1, keepdim=True)

                # Append predicted token to sequence
                trg_seq = torch.cat([trg_seq, next_token], dim=1)

                # Check if <eos> token is predicted for any sequence in the batch
                generated_eos |= (next_token.squeeze(-1) == eos_token_id)

                # Stop early if all sequences have generated <eos> token
                if generated_eos.all():
                    break

            # Ensure that the output is not empty
            if trg_seq.shape[1] == 1:
                trg_seq = torch.tensor([[unk_token_id]], dtype=torch.long,
                                       device=src.device).expand(batch_size, 2)

            return trg_seq
