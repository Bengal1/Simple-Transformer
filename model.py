import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """

    """

    def __init__(self, num_in, num_out):
        super(FeedForward, self).__init__()
        self.hidden_dim = 1000

        self.fc1 = nn.Linear(num_in, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, num_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        fc1_out = self.fc1(x)
        fc2_out = self.fc2(fc1_out)
        return self.relu(fc2_out)


class NormLayer(nn.Module):
    """
    x' = (x - μ)/sqrt(σ^2 + ε)

    --> y = γ·x' + β

    """

    def __init__(self, d_model, epsilon=1e-15):
        super(NormLayer, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.epsilon = epsilon

    def forward(self, x):
        # set statistical parameters
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)

        y = (x - mean) / torch.sqrt(var + self.epsilon)
        return self.gamma * y + self.beta


class MultiHeadAttention(nn.Module):
    """
    --- Self-Attention ---
    Input: X ∈ M×N

    X·W_q = Q ∈ M×d - query matrix
    X·W_k = K ∈ M×d - key matrix
    X·W_v = V ∈ M×d_v - value matrix

    Attention(Q,K,V) =Softmax(Q·K^T/sqrt(d))·V = ΔX' ∈ M×d_v
    ΔX'·W_out = ΔX ∈ M×N
    --> Z = ΔX + X - residual connection

    --- Cross-Attention ---
    Input: X ∈ M×N, Y ∈ L×N

    X·W_q = Q ∈ M×d - query matrix
    Y·W_k = K ∈ L×d - key matrix
    Y·W_v = V ∈ L×d_v - value matrix

    Attention(Q,K,V) =Softmax(Q·K^T/sqrt(d))·V = ΔX ∈ M×d_v
    ΔX'·W_out = ΔX ∈ M×N
    --> Z = ΔX + X - residual connection
    """

    def __init__(self, seq_len, d_k, d_v, num_heads, embed_dim_src, cross_attn=False,
                 batch_first=True, attn_mask=False):
        super(MultiHeadAttention, self).__init__()
        self.cross_attn = cross_attn
        self.attn_mask = attn_mask

        self.w_q = nn.Linear(embed_dim_src, d_k)
        self.w_k = nn.Linear(embed_dim_src, d_k)
        self.w_v = nn.Linear(embed_dim_src, d_v)
        self.w_out = nn.Linear(d_v, embed_dim_src)
        self.attention = nn.MultiheadAttention(embed_dim_src, num_heads, batch_first=batch_first)

    def forward(self, x, **args):
        query = self.w_q(x)
        if self.cross_attn:
            y = args['y']
            key = self.w_k(y)
            value = self.w_v(y)
        else:
            key = self.w_k(x)
            value = self.w_v(x)
        if self.attn_mask:
            delta_x_ = self.attention(query,key,value, attn_mask=0)
        else:
            delta_x_ = self.attention(query, key, value)
        delta_x = self.w_out(delta_x_)

        return x + delta_x


class SimpleTransformer(nn.Module):
    """
    Transformer architecture according to article 'Attention Is All You Need'.

    """

    def __init__(self, embed_dim_src, embed_dim_target, seq_len, d_k=10, d_v=10, num_heads=8):
        super(SimpleTransformer, self).__init__()
        self.h = num_heads

        # Decoder
        self.attention_decoder = MultiHeadAttention(seq_len, d_k, d_v, num_heads, embed_dim_src)
        self.norm1_dec = NormLayer()
        self.ff1 = FeedForward(embed_dim_src,embed_dim_src)
        self.norm2_dec = NormLayer()

        # Encoder
        self.attention_masked_enc = MultiHeadAttention(seq_len, d_k, d_v, num_heads, embed_dim_target, attn_mask=True)
        self.norm1_enc = NormLayer()
        self.attention_cross_enc = MultiHeadAttention(seq_len, d_k, d_v, num_heads, embed_dim_target, cross_attn=True)
        self.norm2_enc = NormLayer()
        self.ff2 = FeedForward(embed_dim_target,embed_dim_target)
        self.norm3_enc = NormLayer()

        # output
        self.w_o = nn.Linear(d_v*self.h, embed_dim_target)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src_embed, target_embed):
        # Decoder
        attn_dec = src_embed + self.attention_decoder.forward(src_embed)
        norm1_d_out = self.norm1_dec(attn_dec)
        ff1_out = attn_dec + self.ff1(norm1_d_out)
        norm2_d_out = self.norm2_dec(ff1_out)

        # Encoder
        attn_masked = target_embed + self.attention_masked_enc_enc.forward(target_embed, attn_mask=0)
        norm1_e_out = self.norm1_enc(attn_masked)
        attn_cross = attn_masked + self.attention_enc.forward(norm1_e_out, y=norm2_d_out)
        norm2_e_out = self.norm2_enc(attn_cross)
        ff2_out = attn_cross + self.ff2(norm2_e_out)
        norm3_e_out = self.norm3_enc(ff2_out)

        # Output block
        out = self.w_o(norm3_e_out)
        probabilities = self.softmax(out)

        return probabilities
