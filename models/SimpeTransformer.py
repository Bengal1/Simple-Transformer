import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Adds positional encoding to the input tensor.The positional encoding is based
    on the formula from 'Attention Is All You Need'.

    Attributes:
        embed_dim (int): The embedding dimension of the model.
        n (int, optional): The base for the sinusoidal encoding. Default is 10000.
    """

    def __init__(self, embed_dim: int, n: int = 10000):
        super().__init__()
        self.embed_dim = embed_dim
        self.n = n

    def _create_positional_encoding(self, seq_len: int,  device: torch.device) -> torch.tensor:
        """Creates the positional encoding matrix.

        Args:
            seq_len (int): Length of the sequence for positional encoding.
            device (torch.device): Device where the tensor should be allocated.

        Returns:
            Tensor: The positional encoding matrix of shape (seq_len, embed_dim).
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
            x (Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            Tensor: Tensor with added positional encoding.
        """
        batch_size, seq_len, _ = x.shape
        pos_encoding = self._create_positional_encoding(seq_len, x.device)

        return x + pos_encoding.unsqueeze(0)  # Broadcast across batch

class FeedForward(nn.Module):
    """
    A FeedForward neural network as used in the Transformer model.

    Attributes:
        d_model (int): The dimension of the input and output embeddings.
        hidden_dim (int, optional): The hidden dimension in the feed-forward network, default is 512.
        dropout (float, optional): The dropout probability, default is 0.1.
    """

    def __init__(self, d_model: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the FeedForward network.
        Applies two linear transformations with a ReLU activation and dropout in between.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class NormLayer(nn.Module):
    """
    Implements layer normalization used in the Transformer.

    Attributes:
        d_model (int): The dimension of the input tensor.
        epsilon (float, optional): A small value added to variance for numerical stability, default is 1e-15.
    """

    def __init__(self, d_model: int, epsilon: float = 1e-15):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies layer normalization.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor: Normalized tensor of the same shape as input.
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        normalized = (x - mean) / torch.sqrt(var + self.epsilon)
        return self.gamma * normalized + self.beta


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module.

    This module implements the multi-head attention mechanism used in Transformer models.
    It supports both self-attention and cross-attention.

    Attributes:
        embed_dim (int): Dimension of input embeddings.
        num_heads (int): Number of attention heads, default is 1.
        d_k (int): Dimension of key vectors per head, default is 64.
        d_v (int): Dimension of value vectors per head, default is 128.
        cross_attn (bool, optional): If True, performs cross-attention (decoder), Defaults is False.
        masked_attn (bool, optional): If True, applies a causal mask for autoregressive decoding, Defaults is False.
    """

    def __init__(self, embed_dim: int, num_heads: int = 1, d_k: int = 64,
                 d_v: int = 128, cross_attn: bool = False, masked_attn: bool = False):
        super().__init__()

        # assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_k = d_k  # size of the key dimension
        self.d_v = d_v  # size of the value dimension

        self.cross_attn = cross_attn
        self.masked_attn = masked_attn

        # Linear projections for Q, K, V for each head
        self.w_q = nn.ModuleList([nn.Linear(embed_dim, d_k) for _ in range(num_heads)])
        self.w_k = nn.ModuleList([nn.Linear(embed_dim, d_k) for _ in range(num_heads)])
        self.w_v = nn.ModuleList([nn.Linear(embed_dim, d_v) for _ in range(num_heads)])
        self.w_out = nn.Linear(d_v * num_heads, embed_dim)

    def _scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                     mask: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        """Computes scaled dot-product attention.

        Args:
            Q (torch.Tensor): Query tensor of shape (batch_size, num_heads, src_len, d_k).
            K (torch.Tensor): Key tensor of shape (batch_size, num_heads, tgt_len, d_k).
            V (torch.Tensor): Value tensor of shape (batch_size, num_heads, tgt_len, d_v).
            mask (torch.Tensor, optional): Attention mask of shape (1, 1, src_len, tgt_len), with -inf for masked positions.

        Returns:
            torch.Tensor: Attention output of shape (batch_size, num_heads, src_len, d_v).
        """
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=Q.dtype, device=Q.device))

        if mask is not None:
            attn_scores = attn_scores + mask

        attn_probs = F.softmax(attn_scores, dim=-1)
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

        # Create dynamic mask of shape (1, 1, M, L)
        mask = None
        if self.masked_attn:
            mask = torch.triu(torch.full((src_len, trg_len), float('-inf'), device=x.device), diagonal=1)
            mask = mask.unsqueeze(0).unsqueeze(1)  # (1, 1, src_len, trg_len)

        # Apply scaled dot-product attention
        attention_output, _ = self._scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads and project to output dimension
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, src_len, -1)
        return self.w_out(attention_output)


class SimpleTransformer(nn.Module):
    """
    Implements the full Transformer model for sequence-to-sequence tasks.

    Attributes:
        src_vocab_size (int): Vocabulary size for the source language.
        trg_vocab_size (int): Vocabulary size for the target language.
        embed_dim (int): The embedding dimension for both source and target embeddings.
        num_heads (int, optional): Number of attention heads, default is 8.
        d_k (int, optional): The dimension of the query and key vectors in multi-head attention, default is 32.
        d_v (int, optional): The dimension of the value vectors in multi-head attention, default is 64.
    """

    def __init__(self, src_vocab_size: int, trg_vocab_size: int, embed_dim:
                    int,num_heads: int = 8, d_k: int = 32, d_v: int = 64):
        super().__init__()

        # Encoder components
        self.embedding_encoder = nn.Embedding(src_vocab_size, embed_dim, padding_idx=1)
        self.positional_encoding_encoder = PositionalEncoding(embed_dim)

        self.attention_encoder = MultiHeadAttention(embed_dim, num_heads, d_k, d_v)
        self.norm1_enc = NormLayer(embed_dim)
        self.ff_enc = FeedForward(embed_dim)
        self.norm2_enc = NormLayer(embed_dim)

        # Decoder components
        self.embedding_decoder = nn.Embedding(trg_vocab_size, embed_dim, padding_idx=1)
        self.positional_encoding_decoder = PositionalEncoding(embed_dim)

        self.attention_masked_dec = MultiHeadAttention(embed_dim, num_heads, d_k, d_v, masked_attn=True)
        self.norm1_dec = NormLayer(embed_dim)
        self.attention_cross_dec = MultiHeadAttention(embed_dim, num_heads, d_k, d_v, cross_attn=True)
        self.norm2_dec = NormLayer(embed_dim)
        self.ff_dec = FeedForward(embed_dim)
        self.norm3_dec = NormLayer(embed_dim)

        # Output layer
        self.w_o = nn.Linear(embed_dim, trg_vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
        """Forward pass for the SimpleTransformer model.

        Args:
            src (Tensor): Source tensor of shape (batch_size, src_seq_len).
            trg (Tensor): Target tensor of shape (batch_size, trg_seq_len).

        Returns:
            Tensor: Output tensor of shape (batch_size, trg_seq_len, trg_vocab_size).
        """

        # Embedding and positional encoding
        src_embed = self.embedding_encoder(src)
        src_pe = self.positional_encoding_encoder(src_embed)

        trg_embed = self.embedding_decoder(trg)
        trg_pe = self.positional_encoding_decoder(trg_embed)

        # Encoder
        attn_enc = self.attention_encoder(src_pe)
        norm1_e_out = self.norm1_enc(attn_enc + src_pe)
        ff_e_out = self.ff_enc(norm1_e_out)
        enc_out = self.norm2_enc(ff_e_out + norm1_e_out)

        # Decoder
        attn_masked_out = self.attention_masked_dec(trg_pe)
        norm1_d_out = self.norm1_dec(trg_pe + attn_masked_out)
        attn_cross_dec = self.attention_cross_dec(norm1_d_out, enc_out)
        norm2_d_out = self.norm2_dec(norm1_d_out + attn_cross_dec)
        ff_d_out = self.ff_dec(norm2_d_out)
        dec_out = self.norm3_dec(norm2_d_out + ff_d_out)

        # Output
        out = self.w_o(dec_out)
        return self.softmax(out)

    def translate(self, src: torch.Tensor) -> torch.Tensor:
        """Translates a given source sequence into the target language using greedy decoding.

        Args:
            src (torch.Tensor): Source tensor of shape (batch_size, src_seq_len).

        Returns:
            torch.Tensor: Predicted target sequence of shape (batch_size, trg_seq_len).
        """
        with torch.no_grad():
            bos_token_id, eos_token_id = 2, 3  # <bos> and <eos> token IDs
            batch_size, max_target_length = src.shape  # Extract input dimensions

            # Encoder
            src_embed = self.embedding_encoder(src)
            src_pe = self.positional_encoding_encoder(src_embed)
            attn_enc = self.attention_encoder(src_pe)
            norm1_enc = self.norm1_enc(attn_enc + src_pe)
            ff_enc = self.ff_enc(norm1_enc)
            enc_out = self.norm2_enc(ff_enc + norm1_enc)  # Final encoder output

            # Decoder
            trg_seq = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=src.device)

            for _ in range(max_target_length):
                trg_embed = self.embedding_decoder(trg_seq)
                trg_pe = self.positional_encoding_decoder(trg_embed)
                attn_masked_dec = self.attention_masked_dec(trg_pe)
                norm1_dec = self.norm1_dec(attn_masked_dec + trg_pe)
                attn_cross_dec = self.attention_cross_dec(norm1_dec, enc_out)
                norm2_dec = self.norm2_dec(attn_cross_dec + norm1_dec)
                ff_dec = self.ff_dec(norm2_dec)
                dec_out = self.norm3_dec(ff_dec + norm2_dec)

                out_logits = self.w_o(dec_out[:, -1, :])
                next_token = out_logits.argmax(dim=-1, keepdim=True)
                trg_seq = torch.cat([trg_seq, next_token], dim=1)

                if (next_token == eos_token_id).all():
                    break

            return trg_seq
