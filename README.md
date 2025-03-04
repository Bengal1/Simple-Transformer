# SimpleTransformer

Transformer architecture according to article 'Attention Is All You Need'.
## Data

```
sentence = "this is a sentence from the dataset" ----> sentence_tokenized = ["this", "is", "a", "sentence", "from", "the", "dataset"]
```


## Transformer

### "Attentiona Is All You Need"

### Attention

**Self-Attention**

$`Input:X∈ℝ^{M×N}`$ 

$`X·W_{q} = Q∈ℝ^{M×d}`$ - *query matrix*<br/>
$`X·W_{k} = K∈ℝ^{M×d}`$ - *key matrix*<br/>
$`X·W_{v} = V∈ℝ^{M×d_v}`$ - *value matrix*<br/>
```math
Attention(Q,K,V) = Softmax \Bigg(\frac{Q K^{T}}{\sqrt{d}} \Bigg)·V = ΔX'∈ℝ^{M×d_v}
```
$`→ ΔX'·W_{out} = ΔX∈ℝ^{M×N}`$<br/>
$`⇨ Y = ΔX + X`$ - *Residual connection*<br/>

**Cross-Attention**

$`Input:  X∈ℝ^{M×N} ,  C∈ℝ^{L×N}`$

$`X·W_q = Q∈ℝ^{M×d}`$ - *query matrix*<br/>
$`C·W_k = K∈ℝ^{L×d}`$ - *key matrix*<br/>
$`C·W_v = V∈ℝ^{L×d_v}`$ - *value matrix*<br/>

```math
Attention(Q,K,V) = Softmax \Bigg(\frac{Q K^{T}}{\sqrt{d}} \Bigg)·V = ΔX'∈ℝ^{M×d_v}
```
$`→ ΔX'·W_{out} = ΔX∈ℝ^{M×N}`$<br/>
$`⇨ Y = ΔX + X`$ - *Residual connection*

```math
Attention(Q,K,V) = Softmax \Bigg(\frac{Q K^{T}}{\sqrt{d}} \Bigg)·V
```

### FeedForward

### Embedding

### Positional Encodeing

### Normalizing
