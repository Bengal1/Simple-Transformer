# SimpleTransformer

Transformer architecture according to article 'Attention Is All You Need'.


## Attention

*Self-Attention*

Input: X ∈ M×N 

$`X·W_{q} = Q ∈ M×d`$ - *query matrix*<br/>
$`X·W_{k} = K ∈ M×d`$ - *key matrix*<br/>
$`X·W_{v} = V ∈ M×d_{v}`$ - *value matrix*<br/>

$`Attention(Q,K,V) = Softmax(\frac{Q K^{T}}{\sqrt{d}})·V = ΔX' ∈ M×d_v`$<br/>
$`→ ΔX'·W_{out} = ΔX ∈ M×N`$<br/>
$`⇨ Y = ΔX + X`$ - *Residual connection*<br/>

*Cross-Attention*

$`Input:  X∈R^{M×N} ,  C∈R^{L×N}`$

$`X·W_q = Q ∈ M×d`$ - *query matrix*<br/>
$`C·W_k = K ∈ L×d`$ - *key matrix*<br/>
$`C·W_v = V ∈ L×d_v`$ - *value matrix*<br/>

```math
Attention(Q,K,V) = Softmax(\frac{Q K^{T}}{\sqrt{d}})·V = ΔX' ∈ M×d_v
```
$`→ ΔX'·W_{out} = ΔX ∈ M×N`$<br/>
$`⇨ Y = ΔX + X`$ - residual connection

```math
Attention(Q,K,V) = Softmax(\frac{Q K^{T}}{\sqrt{d}})·V
```
