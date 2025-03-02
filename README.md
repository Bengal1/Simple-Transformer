# SimpleTransformer

Transformer architecture according to article 'Attention Is All You Need'.


## Attention

*Self-Attention*
Input: X ∈ M×N 

$`X·W_{q} = Q ∈ M×d`$ *- query matrix*<br/>
$`X·W_{k} = K ∈ M×d`$ *- key matrix*<br/>
$`X·W_{v} = V ∈ M×d_{v}`$ *- value matrix*<br/>

$`Attention(Q,K,V) =Softmax(Q·K^T/sqrt(d))·V = ΔX' ∈ M×d_v`$
ΔX'·W_out = ΔX ∈ M×N
--> Z = ΔX + X - Residual connection
*Cross-Attention*
```math
Input: X ∈ M×N, Y ∈ L×N

X·W_q = Q ∈ M×d - query matrix
Y·W_k = K ∈ L×d - key matrix
Y·W_v = V ∈ L×d_v - value matrix

Attention(Q,K,V) =Softmax(Q·K^T/sqrt(d))·V = ΔX ∈ M×d_v
ΔX'·W_out = ΔX ∈ M×N
--> Z = ΔX + X - residual connection
```
```math
Attention(Q,K,V) = Softmax(\frac{Q K^{T}}{\sqrt{d}})V
```
