import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Positional Encoding:
    PE(k,2i) = sin(k/n**(2i/d))
    PE(k,2i+1) = cos(k/n**(2i/d))

    d : Dimension of the model output (output embedding space).
    k : Position of an object in the input sequence, 0 <= k < M; M=sequence length.
    n : User defined scalar, set to 10,000 in 'Attention Is All You Need'.
    PE(k,j) : Positional encoding of the j-th index in the k-th object in the input sequence.

    Input: accept input of size(N,M,d_embedding) or size(M,d_embedding), when N=batch size.
    Output: return the same size Tensor with positional encoding.
    """

    def __init__(self, seq_len, d_model, n=10000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.n = n
        self.pos_encoding = torch.zeros(seq_len, d_model)

        k_pos = torch.arange(0, seq_len).unsqueeze(dim=1).float()
        _2i = torch.arange(0, self.d_model, step=2).float()

        self.pos_encoding[:, 0::2] += torch.sin(k_pos / self.n ** (_2i / self.d_model))
        self.pos_encoding[:, 1::2] += torch.cos(k_pos / self.n ** (_2i / self.d_model))

    def forward(self, x):

        x[:] += self.pos_encoding

        return x

class FeedForward(nn.Module):
    def __init__(self, d_model, hiden_dim=2048, dropout=0.1):
        """
        d_model: the dimension of input and output (typically 512 or 1024)
        hidden_dim: the inner dimension of the feed-forward network (typically 2048)
        dropout: dropout probability (default: 0.1)

        Input: X ∈ M×N
        """
        super(FeedForward, self).__init__()

        self.fc1 = nn.Linear(d_model, hiden_dim)
        self.fc2 = nn.Linear(hiden_dim, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class NormLayer(nn.Module):
    """
    Layer normalization: x' = (x - μ) / sqrt(σ^2 + ε)
    Then apply scaling (gamma) and shifting (beta) parameters.
    --> y = γ·x' + β

    Inputs:
        - x: (N, M, E) where N is batch size, M is sequence length, and E is embedding size
        - gamma: scaling parameter (learnable)
        - beta: shifting parameter (learnable)
        - epsilon: small constant added to variance to prevent division by zero

    Outputs:
        - y: normalized output (same shape as x)
    """

    def __init__(self, d_model, epsilon=1e-15):
        super(NormLayer, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.epsilon = epsilon

    def forward(self, x):

        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)

        normalized = (x - mean) / torch.sqrt(var + self.epsilon)

        output = self.gamma * normalized + self.beta
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, max_length, embed_dim, num_heads=8, cross_attn=False, masked_attn=False):
        super(MultiHeadAttention, self).__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.cross_attn = cross_attn
        self.masked_attn = masked_attn
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads  # Dimension per head

        # Linear projections for Q, K, V
        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)
        self.w_out = nn.Linear(embed_dim, embed_dim)

        # Mask for decoder's self-attention
        if masked_attn:
            mask = torch.triu(torch.ones(1, 1, max_length, max_length, dtype=torch.bool), diagonal=1)
            self.register_buffer("mask", mask)


    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Q, K, V: (batch_size, num_heads, max_length, d_k)
        mask: (1, 1, max_length, max_length) (broadcastable)
        """
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=Q.dtype, device=Q.device))

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_probs, V), attn_probs  # (batch_size, num_heads, max_length, d_k)

    def forward(self, x, y=None):
        """
        x: (batch_size, max_length, embed_dim)  - always used
        y: (batch_size, max_length, embed_dim)  - used in cross-attention (decoder)
        """

        batch_size, max_length, _ = x.shape

        # Compute Q, K, V
        K, V = self.w_k(y if self.cross_attn and y is not None else x), \
               self.w_v(y if self.cross_attn and y is not None else x)
        Q = self.w_q(x)

        # Reshape to (batch_size, num_heads, max_length, d_k)
        Q = Q.view(batch_size, max_length, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, max_length, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, max_length, self.num_heads, self.d_k).transpose(1, 2)

        # Ensure mask has the right shape
        mask = self.mask if self.masked_attn else None
        attention_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads and project
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, max_length, -1)
        return self.w_out(attention_output)  # Final projection


class SimpleTransformer(nn.Module):
    """
    Transformer architecture according to article 'Attention Is All You Need'.
    TODO:
        * set d_k and d_v
    """

    def __init__(self, embed_dim, max_length, trg_vocab_size, d_v=10, num_heads=8):
        super(SimpleTransformer, self).__init__()
        self.h = num_heads

        # Encoder
        # self.embedding_encoder = nn.Embedding()
        # self.pe_encoder = PositionalEncoding()
        self.attention_encoder = MultiHeadAttention(max_length, embed_dim, num_heads, embed_dim)
        self.norm1_enc = NormLayer(embed_dim)
        self.ff1 = FeedForward(embed_dim,embed_dim)
        self.norm2_enc = NormLayer(embed_dim)

        # Decoder
        self.attention_masked_dec = MultiHeadAttention(max_length, embed_dim, num_heads, masked_attn=True)
        self.norm1_dec = NormLayer(embed_dim)
        self.attention_cross_dec = MultiHeadAttention(max_length, embed_dim, num_heads, cross_attn=True)
        self.norm2_dec = NormLayer(embed_dim)
        self.ff2 = FeedForward(embed_dim,embed_dim)
        self.norm3_dec = NormLayer(embed_dim)

        # output
        self.w_o = nn.Linear(d_v*self.h, trg_vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src_embed, trg_embed):
        # Encoder
        # embed = self.embeding_encoder(src_embed)
        # pe_enc = self.pe_encoder(embed)
        attn_enc = self.attention_encoder.forward(src_embed)
        norm1_e_out = self.norm1_enc(attn_enc + src_embed)
        ff1_out = self.ff1(norm1_e_out)
        norm2_e_out = self.norm2_dec(ff1_out + norm1_e_out)

        # Decoder
        attn_masked = self.attention_masked_dec.forward(trg_embed)
        norm1_d_out = self.norm1_dec(attn_masked + trg_embed)
        attn_cross = self.attention_dec.forward(norm1_e_out, y=norm2_e_out)
        norm2_d_out = self.norm2_enc(attn_cross + norm1_d_out)
        ff2_out = self.ff2(norm2_d_out)
        norm3_d_out = self.norm3_enc(ff2_out + norm2_d_out)

        # Output block
        # TODO: flatten sentence to fc layer
        out = self.w_o(norm3_d_out)
        probabilities = self.softmax(out)

        return probabilities
