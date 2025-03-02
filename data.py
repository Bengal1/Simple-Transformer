import torch
import torch.nn as nn
import torchtext
from torchtext.vocab import Vocab
from torchtext.data.utils import get_tokenizer



# def get_data():
#     tokenizer_src = get_tokenizer("en")
#     tokenizer_target = get_tokenizer("fr")
#
# class Embedding(nn.Module):
#     def __init__(self):
#         super(Embedding, self).__init__()


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

