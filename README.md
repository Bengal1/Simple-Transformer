# SimpleTransformer Guide

This is a practical guide for building [*Transformer*](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)), and it applies to beginners who like to know how to start building a Transformer with Pytorch. *The Transformer* is a deep learning architecture that was developed by researchers at Google and is based on the multi-head attention mechanism, which was proposed in the 2017 paper "Attention Is All You Need".<br/>
SimpleTransformer architecture is built according to article "Attention Is All You Need", In this project we will use it for [Machine Translation](https://en.wikipedia.org/wiki/Machine_translation)<br/>
This Repository is built for learning purposes, and its goal is to help people who would like to start coding transformer excuting NLP tasks.

## Requirements
* Python 3
* [Pytorch](https://pytorch.org/get-started/locally/)
* [Torchtext](https://pytorch.org/text/stable/index.html)
* [spaCy](https://spacy.io/)

## Data
In this repository we will excute the task of *Machine Translation* french to english. therfore the raw data is a csv file of 2 columns one in english and the other one in french in each row there is a word or sentence in english and its translation in french.

```
                   English words/sentences                             French words/sentences
0                                                     Hi.                                             Salut!
1                                                    Run!                                            Cours !
2                                                    Run!                                           Courez !
3                                                    Who?                                              Qui ?
4                                                    Wow!                                         Ça alors !
...                                                   ...                                                ...
175616  Top-down economics never works, said Obama. "T...  « L'économie en partant du haut vers le bas, ç...
175617  A carbon footprint is the amount of carbon dio...  Une empreinte carbone est la somme de pollutio...
175618  Death is something that we're often discourage...  La mort est une chose qu'on nous décourage sou...
175619  Since there are usually multiple websites on a...  Puisqu'il y a de multiples sites web sur chaqu...
175620  If someone who doesn't know your background sa...  Si quelqu'un qui ne connaît pas vos antécédent...

[175621 rows x 2 columns]
```

123,100 english unique values and 165,975 french unique values [TODO: check if true]

[dataset source](http://www.manythings.org/anki/)
### ****Tokenization****
In order to prepare the data for traning we need tokenization - convet words/sentences to tokens. The computer doesn't know what to do with words. when you feed it the sentence "This Simple Transformer Guide!" it doesn't understand the meaning of the words and the relations between them.<br/>
So what do computer understand? the understand numbers in the core of computer it undrstand binary values($`V_low` and $`V_high`), but on higher levels it understand number and tensors (vectors, matrices, 3D matrices,...) and mathematical relation between them.
In order to provide the computer workable data we decompose the sentence into tokens and covert every token to a dense vector (process called *Embedding*).

```ruby
sentence = "This Simple Transformer Guide!"
⇨ sentence_tokenized = ['This', 'Simple', 'Transformer', 'Guide', '!']
```
Before embedding, we would like to structure the data in such a way that it is easy for the transformer to receive it, so we will define a fixxed length to sentences `max_length`, when we pad sentence that wre shorter (This is the method we used).
* *Altenative method*: use max length 95% of the data. meaning 95% of the data will fit with no problem and 5% will be truncated according to size (the presentage can be changed, for example 90%). This approach allows you to handle the majority of the data, while avoiding excessively long sequences. Sacrificing 10% of data integrity to make the model smaller and more efficient.

In order to give the model contextual sign and mange the data better, we use special tokens
```
special_tokens = ['<unk>', '<pad>', '<bos>', '<eos>']

<unk> - unknown words.
<pad> - use for padding.
<bos> - beginning of sentence.
<eos> - end of sentence.
We sets <unk> as the default. 
```
After sentence tokenization, we put before the sentence the beginning of sentence token, `<bos>`, and after it the end of sentence token, `<eos>`, and pad with padding token, `<pad>`, the remainder of the sentence up to `max_length`.<br/>
The unknown word token ,`<unk>`, use for words that are not in the vocabulary and dealing with failures, and for that reason ee sets `<unk>` as the default. 
```ruby
sentence_tokenized = ['This', 'Simple', 'Transformer', 'Guide', '!']
⇨ sentence_for_embedding[max_length] = ['<bos>','This', 'Simple', 'Transformer', 'Guide', '!', '<eos>', '<pad>',..., '<pad>']
```

### Embedding
* Alternative method to embedding: Token IDs - token IDs -

how embedding work

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


