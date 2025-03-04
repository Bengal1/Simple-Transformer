# SimpleTransformer

Transformer architecture according to article 'Attention Is All You Need'.
## Data
The raw data is a csv file...
### Tokenization
```ruby
sentence = "this is a sentence from the dataset"
⇨ sentence_tokenized = ['this', 'is', 'a', 'sentence', 'from', 'the', 'dataset']

special_tokens = ['<unk>', '<pad>', '<bos>', '<eos>']

sentence_for_train[max_length] = ['<bos>', 'this', 'is', 'a', 'sentence', 'from', 'the', 'dataset', '<eos>', '<pad>',..., '<pad>']
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
alternative = Token IDs

how embedding work

what are the properties of embedding - king - queen, man - woman. gender direction, status(royalty) direction and so on



### Positional Encodeing

$`PE(k,2i) = sin \Bigg(\frac{k}{n^{2i/d}} \Bigg)`$<br/>
$`PE(k,2i+1) = cos \Bigg(\frac{k}{n^{2i/d}} \Bigg)`$<br/>

*k* - 

*n* - 

*d* - 

*i* - 

*PE(k,j)* - 

#### Example:
Lets assume sequence length is M.

$`PE(k=0) = [sin \Bigg(\frac{0}{10,000^{0/d}} \Bigg), cos \Bigg(\frac{0}{10,000^{0/d}} \Bigg), sin \Bigg(\frac{0}{10,000^{2/d}} \Bigg), cos \Bigg(\frac{0}{10,000^{2/d}} \Bigg),..., sin \Bigg(\frac{0}{10,000^{d-2/2d}} \Bigg), cos \Bigg(\frac{0}{10,000^{d-2/2d}} \Bigg)]`$<br/>
$`PE(k=1) = [sin \Bigg(\frac{1}{10,000^{0/d}} \Bigg), cos \Bigg(\frac{1}{10,000^{0/d}} \Bigg), sin \Bigg(\frac{1}{10,000^{2/d}} \Bigg), cos \Bigg(\frac{1}{10,000^{2/d}} \Bigg),..., sin \Bigg(\frac{1}{10,000^{d-2/2d}} \Bigg), cos \Bigg(\frac{1}{10,000^{d-2/2d}} \Bigg)]`$<br/>
.<br/>
.<br/>
.<br/>
$`PE(k=0) = [sin \Bigg(\frac{M-1}{10,000^{0/d}} \Bigg), cos \Bigg(\frac{M-1}{10,000^{0/d}} \Bigg), sin \Bigg(\frac{M-1}{10,000^{2/d}} \Bigg), cos \Bigg(\frac{M-1}{10,000^{2/d}} \Bigg),..., sin \Bigg(\frac{M-1}{10,000^{d-2/2d}} \Bigg), cos \Bigg(\frac{M-1}{10,000^{d-2/2d}} \Bigg)]`$<br/>

### Normalizing

Layer normalization: $`x' = \frac{(x - μ)}{\sqrt{σ^{2} + ε}}`$<br/>
Then apply scaling (gamma) and shifting (beta) parameters.<br/>
⇨  $`y = γ·x' + β`$<br/>

