# SimpleTransformer

he transformer is a deep learning architecture that was developed by researchers at Google and is based on the multi-head attention mechanism, which was proposed in the 2017 paper "Attention Is All You Need".
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

$`X·W_{q} = Q∈ℝ^{M×d_k}`$  -  *query matrix*<br/>
$`X·W_{k} = K∈ℝ^{M×d_k}`$  -  *key matrix*<br/>
$`X·W_{v} = V∈ℝ^{M×d_v}`$  -  *value matrix*<br/>
```math
Attention(Q,K,V) = Softmax \Bigg(\frac{Q K^{T}}{\sqrt{d}} \Bigg)·V = ΔX'∈ℝ^{M×d_v}
```
$`if`$ $`d_v \ne N → W_{out}∈ℝ^{d_v×N}`$<br/>
$`→ ΔX'·W_{out} = ΔX∈ℝ^{M×N}`$<br/>
$`⇨ Y = ΔX + X`$ - *Residual connection*<br/>


TODO: masked attention

**Cross-Attention**

input and conditional input, can be the target output or maybe prompt words

$`Input:  X∈ℝ^{M×N} ,  C∈ℝ^{L×N}`$

$`X·W_q = Q∈ℝ^{M×d_k}`$ - *query matrix*<br/>
$`C·W_k = K∈ℝ^{L×d_k}`$ - *key matrix*<br/>
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

Positional encoding ...

```math
\begin{cases}PE(k,2i) = sin \Bigg(\frac{k}{n^{2i/d}} \Bigg)\\\

PE(k,2i+1) = cos \Bigg(\frac{k}{n^{2i/d}} \Bigg)\end{cases}
```
<br/><br/>
`k` - Position of an object in the input sequence, $`0 \le k <M`$ (M=sequence length).<br/>
`n` - User defined scalar. Set to 10,000 in the article "Attention Is All You Need".<br/>
`d` - Dimension of the model (output or output embedding space).<br/>
`i` - Used for mapping column's/object's indices,  $`0 \le i < \frac{2}{d}`$.<br/>
`PE(k,j)` - Positional encoding of thr j-th index in the k-th object in the input sequence.<br/>

#### Example:
Lets assume sequence length is M (M object/sentences).

$`PE(k=0) = [sin \Bigg(\frac{0}{10,000^{\frac{0}{d}}} \Bigg), cos \Bigg(\frac{0}{10,000^{\frac{0}{d}}} \Bigg), sin \Bigg(\frac{0}{10,000^{\frac{2}{d}}} \Bigg), cos \Bigg(\frac{0}{10,000^{\frac{2}{d}}} \Bigg),..., sin \Bigg(\frac{0}{10,000^{\frac{d-2}{d}}} \Bigg), cos \Bigg(\frac{0}{10,000^{\frac{d-2}{d}}} \Bigg)]`$<br/>
$`PE(k=1) = [sin \Bigg(\frac{1}{10,000^{\frac{0}{d}}} \Bigg), cos \Bigg(\frac{1}{10,000^{\frac{0}{d}}} \Bigg), sin \Bigg(\frac{1}{10,000^{\frac{2}{d}}} \Bigg), cos \Bigg(\frac{1}{10,000^{\frac{2}{d}}} \Bigg),..., sin \Bigg(\frac{1}{10,000^{\frac{d-2}{d}}} \Bigg), cos \Bigg(\frac{1}{10,000^{\frac{d-2}{d}}} \Bigg)]`$<br/>
.<br/>
.<br/>
.<br/>
$`PE(k=M-1) = [sin \Bigg(\frac{M-1}{10,000^{\frac{0}{d}}} \Bigg), cos \Bigg(\frac{M-1}{10,000^{\frac{0}{d}}} \Bigg), sin \Bigg(\frac{M-1}{10,000^{\frac{2}{d}}} \Bigg), cos \Bigg(\frac{M-1}{10,000^{\frac{2}{d}}} \Bigg),..., sin \Bigg(\frac{M-1}{10,000^{\frac{d-2}{d}}} \Bigg), cos \Bigg(\frac{M-1}{10,000^{\frac{d-2}{d}}} \Bigg)]`$<br/>

### Normalizing

Layer normalization:   $`x' = \frac{(x - μ)}{\sqrt{σ^{2} + ε}}`$<br/>
Then apply scaling (gamma) and shifting (beta) parameters.<br/>
⇨  $`y = γ·x' + β`$<br/>


## References
[Attention Is All You Need](https://arxiv.org/abs/1706.03762)


