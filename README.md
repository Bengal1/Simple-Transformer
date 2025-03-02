# SimpleTransformer

Transformer architecture according to article 'Attention Is All You Need'.


## Attention

```math
--- Self-Attention ---
Input: X ∈ M×N <br /><br />

X·W_{q} = Q ∈ M×d - query matrix
X·W_{k} = K ∈ M×d - key matrix
X·W_{v} = V ∈ M×d_{v} - value matrix

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
```
```math
Attention(Q,K,V) = Softmax(\frac{Q K^{T}}{\sqrt{d_{k}}})V
```
