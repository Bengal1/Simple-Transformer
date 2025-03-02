# SimpleTransformer

Transformer architecture according to article 'Attention Is All You Need'.


## Attention

```math
--- Self-Attention --- <br/>
    Input: X ∈ M×N <br /><br />

    X·W_q = Q ∈ M×d - query matrix <br />
    X·W_k = K ∈ M×d - key matrix <br />
    X·W_v = V ∈ M×d_v - value matrix <br />

    Attention(Q,K,V) =Softmax(Q·K^T/sqrt(d))·V = ΔX' ∈ M×d_v <br />
    ΔX'·W_out = ΔX ∈ M×N <br />
    --> Z = ΔX + X - Residual connection <br />

    --- Cross-Attention ---
    Input: X ∈ M×N, Y ∈ L×N

    X·W_q = Q ∈ M×d - query matrix
    Y·W_k = K ∈ L×d - key matrix
    Y·W_v = V ∈ L×d_v - value matrix

    Attention(Q,K,V) =Softmax(Q·K^T/sqrt(d))·V = ΔX ∈ M×d_v
    ΔX'·W_out = ΔX ∈ M×N
    --> Z = ΔX + X - residual connection


Attention(Q,K,V) = Softmax(\frac{Q K^{T}}{\sqrt{d_{k}}})V
```
