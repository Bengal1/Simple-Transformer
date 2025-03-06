import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """
    Adds positional encoding to the input tensor. The positional encoding is
    based on the formula from 'Attention Is All You Need'.

    Parameters:
    - seq_len (int): The length of the sequence.
    - d_model (int): The dimension of the model's embeddings.
    - n (int, optional): The scaling factor for the encoding, default is 10000.

    Input:
    - x (Tensor): Tensor of shape (batch_size, seq_len, d_model)

    Output:
    - Tensor: The same input tensor with added positional encoding.
    """

    def __init__(self, seq_len: int, d_model: int, n: int = 10000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.n = n
        self.pos_encoding = self._create_positional_encoding(seq_len)

    def _create_positional_encoding(self, seq_len: int):
        """
        Creates the positional encoding matrix.

        Parameters:
        - seq_len (int): Length of the sequence for positional encoding.

        Returns:
        - Tensor: The positional encoding matrix of shape (seq_len, d_model).
        """
        pos_encoding = torch.zeros(seq_len, self.d_model)
        k_pos = torch.arange(0, seq_len).unsqueeze(dim=1).float()
        _2i = torch.arange(0, self.d_model, step=2).float()

        pos_encoding[:, 0::2] = torch.sin(k_pos / self.n ** (_2i / self.d_model))
        pos_encoding[:, 1::2] = torch.cos(k_pos / self.n ** (_2i / self.d_model))
        return pos_encoding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the input tensor.

        Parameters:
        - x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
        - Tensor: Tensor with added positional encoding.
        """
        x += self.pos_encoding
        return x


class FeedForward(nn.Module):
    """
    A FeedForward neural network as used in the Transformer model.

    Parameters:
    - d_model (int): The dimension of the input and output embeddings.
    - hidden_dim (int, optional): The hidden dimension in the feed-forward network, default is 2048.
    - dropout (float, optional): The dropout probability, default is 0.1.
    """

    def __init__(self, d_model: int, hidden_dim: int = 2048, dropout: float = 0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the FeedForward network.

        Parameters:
        - x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
        - Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class NormLayer(nn.Module):
    """
    Implements layer normalization used in the Transformer.

    Parameters:
    - d_model (int): The dimension of the input tensor.
    - epsilon (float, optional): A small value added to variance for numerical stability, default is 1e-15.
    """

    def __init__(self, d_model: int, epsilon: float = 1e-15):
        super(NormLayer, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies layer normalization.

        Parameters:
        - x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
        - Tensor: Normalized tensor of the same shape as input.
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        normalized = (x - mean) / torch.sqrt(var + self.epsilon)
        return self.gamma * normalized + self.beta


class MultiHeadAttention(nn.Module):
    """
    Implements the Multi-Head Attention mechanism used in the Transformer model.

    Parameters:
    - max_length (int): The maximum length of the sequence.
    - embed_dim (int): The embedding dimension for the input and output.
    - num_heads (int): The number of attention heads.
    - cross_attn (bool, optional): Whether this is a cross-attention layer (decoder), default is False.
    - masked_attn (bool, optional): Whether this is a masked attention layer (for causal attention), default is False.
    """

    def __init__(self, max_length: int, embed_dim: int, num_heads: int = 8, cross_attn: bool = False, masked_attn: bool = False):
        super(MultiHeadAttention, self).__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads

        self.cross_attn = cross_attn
        self.masked_attn = masked_attn

        # Linear projections for Q, K, V
        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)
        self.w_out = nn.Linear(embed_dim, embed_dim)

        if masked_attn:
            mask = torch.triu(torch.ones(1, 1, max_length, max_length, dtype=torch.bool), diagonal=1)
            self.register_buffer("mask", mask)

    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Computes the scaled dot-product attention.

        Parameters:
        - Q (Tensor): Query tensor of shape (batch_size, num_heads, max_length, d_k)
        - K (Tensor): Key tensor of shape (batch_size, num_heads, max_length, d_k)
        - V (Tensor): Value tensor of shape (batch_size, num_heads, max_length, d_k)
        - mask (Tensor, optional): Mask tensor for masking certain positions, default is None.

        Returns:
        - Tensor: Attention output tensor.
        """
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=Q.dtype, device=Q.device))

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_probs, V), attn_probs

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for multi-head attention.

        Parameters:
        - x (Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)
        - y (Tensor, optional): Tensor for cross-attention, default is None.

        Returns:
        - Tensor: Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, max_length, _ = x.shape

        # Compute Q, K, V
        K, V = self.w_k(y if self.cross_attn and y is not None else x), self.w_v(y if self.cross_attn and y is not None else x)
        Q = self.w_q(x)

        # Reshape Q, K, V for multi-head attention
        Q = Q.view(batch_size, max_length, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, max_length, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, max_length, self.num_heads, self.d_k).transpose(1, 2)

        mask = self.mask if self.masked_attn else None
        attention_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads and project to output dimension
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, max_length, -1)
        return self.w_out(attention_output)


class SimpleTransformer(nn.Module):
    """
    Implements the full Transformer model for sequence-to-sequence tasks.

    Parameters:
    - src_vocab_size (int): Vocabulary size for the source language.
    - trg_vocab_size (int): Vocabulary size for the target language.
    - embed_dim (int): The embedding dimension for both source and target embeddings.
    - max_length (int): Maximum sequence length.
    - num_heads (int): Number of attention heads.
    - d_v (int, optional): The dimension of the value vectors in multi-head attention, default is 10.
    """

    def __init__(self, src_vocab_size: int, trg_vocab_size: int, embed_dim: int, max_length: int, num_heads: int = 8, d_v: int = 10):
        super(SimpleTransformer, self).__init__()

        # Encoder components
        self.embedding_encoder = nn.Embedding(src_vocab_size, embed_dim)
        self.positional_encoding_encoder = PositionalEncoding(max_length, embed_dim)

        # Decoder components
        self.embedding_decoder = nn.Embedding(trg_vocab_size, embed_dim)
        self.positional_encoding_decoder = PositionalEncoding(max_length, embed_dim)

        # Attention and FeedForward layers
        self.attention_encoder = MultiHeadAttention(max_length, embed_dim, num_heads)
        self.norm1_enc = NormLayer(embed_dim)
        self.ff1 = FeedForward(embed_dim)
        self.norm2_enc = NormLayer(embed_dim)

        self.attention_masked_dec = MultiHeadAttention(max_length, embed_dim, num_heads, masked_attn=True)
        self.norm1_dec = NormLayer(embed_dim)
        self.attention_cross_dec = MultiHeadAttention(max_length, embed_dim, num_heads, cross_attn=True)
        self.norm2_dec = NormLayer(embed_dim)
        self.ff2 = FeedForward(embed_dim)
        self.norm3_dec = NormLayer(embed_dim)

        # Output layer
        self.w_o = nn.Linear(d_v * num_heads, trg_vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SimpleTransformer model.

        Parameters:
        - src (Tensor): Source tensor of shape (batch_size, src_seq_len).
        - trg (Tensor): Target tensor of shape (batch_size, trg_seq_len).

        Returns:
        - Tensor: Output tensor of shape (batch_size, trg_seq_len, trg_vocab_size).
        """
        # Embedding and positional encoding
        src_embed = self.embedding_encoder(src)
        src_embed = self.positional_encoding_encoder(src_embed)

        trg_embed = self.embedding_decoder(trg)
        trg_embed = self.positional_encoding_decoder(trg_embed)

        # Encoder
        attn_enc = self.attention_encoder(src_embed)
        norm1_e_out = self.norm1_enc(attn_enc + src_embed)
        ff1_out = self.ff1
