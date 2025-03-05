# SimpleTransformer Guide
<img align="right" width="200"  src="https://github.com/user-attachments/assets/e55c4e75-3ed1-4b12-95d6-49bdf9dc10a6">

This is a practical guide for building [*Transformer*](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)), and it applies to beginners who like to know how to start building a Transformer with Pytorch. *The Transformer* is a deep learning architecture that was developed by researchers at Google and is based on the multi-head attention mechanism, which was proposed in the 2017 paper "Attention Is All You Need".<br/>
SimpleTransformer architecture is built according to article "Attention Is All You Need", In this project we will use it for [Machine Translation](https://en.wikipedia.org/wiki/Machine_translation). <br/>
This Repository is built for learning purposes, and its goal is to help people who would like to start coding transformer excuting [*NLP (Natural Language Processing)*](https://en.wikipedia.org/wiki/Natural_language_processing) tasks.

## Requirements
* Python 3
* [Pytorch](https://pytorch.org/get-started/locally/)
* [Torchtext](https://pytorch.org/text/stable/index.html)
* [spaCy](https://spacy.io/)

## Data
TODO: 
* check if unique value number correct
* more detailes on sizes and staff  

In this repository we will excute the task of *Machine Translation* french to english. therfore the raw data is a csv file of 2 columns one in english and the other one in french in each row there is a word or sentence in english and its translation in french.

```
       |            English words/sentences                |             French words/sentences                |
       |---------------------------------------------------|---------------------------------------------------|
0      | Hi.                                               | Salut!                                            |
1      | Run!                                              | Cours !                                           |
2      | Run!                                              | Courez !                                          |
3      | Who?                                              | Qui ?                                             |
4      | Wow!                                              | Ça alors !                                        |
...    | ...                                               | ...                                               |
175616 | Top-down economics never works, said Obama. "T... | « L'économie en partant du haut vers le bas, ç... |
175617 | A carbon footprint is the amount of carbon dio... | Une empreinte carbone est la somme de pollutio... |
175618 | Death is something that we're often discourage... | La mort est une chose qu'on nous décourage sou... |
175619 | Since there are usually multiple websites on a... | Puisqu'il y a de multiples sites web sur chaqu... |
175620 | If someone who doesn't know your background sa... | Si quelqu'un qui ne connaît pas vos antécédent... |

[175621 rows x 2 columns]
```

Containing 123,100 unique english values and 165,975 unique french values.

[dataset source](http://www.manythings.org/anki/)
### Tokenization
In order to prepare the data for traning we need tokenization - convet words/sentences to tokens. The computer doesn't know what to do with words. when you feed it the sentence "This Simple Transformer Guide!" it doesn't understand the meaning of the words and the relations between them.<br/>
So what do computer understand? the understand numbers in the core of computer it undrstand binary values($`V_low` and $`V_high`), but on higher levels it understand number and tensors (vectors, matrices, 3D matrices,...) and mathematical relation between them.
In order to provide the computer workable data we decompose the sentence into tokens and covert every token to a dense vector (process called *Embedding*).

```ruby
sentence = "This Simple Transformer Guide!"
⇨ sentence_tokenized = ['This', 'Simple', 'Transformer', 'Guide', '!']
```
Before embedding, we would like to structure the data in such a way that it is easy for the transformer to receive it, so we will define a fixed length to sentences `max_length`, and then we pad sentence that are shorter (This is the method in use here).
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
The unknown word token ,`<unk>`, use for words that are not in the vocabulary and dealing with failures, and for that reason we sets `<unk>` as the default. 
```ruby
sentence_tokenized = ['This', 'Simple', 'Transformer', 'Guide', '!']
⇨ sentence_for_embedding[max_length] = ['<bos>','This', 'Simple', 'Transformer', 'Guide', '!', '<eos>', '<pad>',..., '<pad>']
```

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Embedding

Embeddings are representations of values or objects like text, images, and audio that are designed to be consumed by machine learning models and semantic search algorithms. Embeddings translate objects like these into a mathematical form according to the preset factors, enable machine learning models to interact with various data types. 
In our case we get a tokenized sequence (sentence, `M=max_length`) and we convert every token to a vector in the $`ℝ^{E}`$(`E=embedding_dim`, for more information on [*Real Vector Space*](https://en.wikipedia.org/wiki/Real_coordinate_space)) and we get for every sequence a matrix of size $`ℝ^{M×E}`$.

#### Intuitive understanding of Embedding

<img align="right" width="500"  src="https://github.com/user-attachments/assets/edf0a13e-fa50-4dbd-a040-940fcf3c0d76">

This explaination is for intuitive understanding of Embedding, you will need basic vector analysis to best understand it.<br/>
Lets assume we have the tokens `{'king', 'queen', 'man', 'woman'}` and we covnert them to embedding vectors: $`\Big\{ e_{king}, e_{queen}, e_{man}, e_{woman} \Big\}`$, So for example we would exapect, for good embedding, the next mathematical semantic connection:
```math
e_{king} - e_{queen} = e_{man} - e_{woman}
```

And we can interpret it as the gender difference between the vectors, meaning in the $`ℝ^{E}`$ embedding space (Lets assume E is big), there is an direction of gender, the more manly attrubutes the token has the further the the vector will go in that direction and the same for womaly attributes in the opposite direction. 
We can also look at this mathematical semantic connection: 
```math
e_{king} - e_{man} = e_{queen} - e_{woman}
```
We can interpret it as if we strip the king from his gender then the vector that we get is the status/Royal vector as well as for the queen, meaning a royal direction.<br/>
And also it expected to get from the king vector to the queen vector we will do: 
<br/>
&emsp;&emsp;&emsp;&emsp;&emsp; $`e_{king} - e_{man} + e_{woman} = e_{queen}`$
<br/><br/>

#### How Can $`ℝ^{E}`$ Holds Language Semantics?

In Reality that is not what happenning. There is no equality in the mathematical connection, probably because there is more for king part to gender and royalty, but a rough axis direction can be noticed. We can interpret that for a some large vocabulary and $`ℝ^{E}`$, large embedding space, there will be semantic direction in this space. We expect them to be orthogonal, so that an object in this space when getting shifted in the 'Royal' direction it would not be shifted in unrelated direction like 'Size', 'Metallic' etc, meanning larger the embedding space the more semantics it can hold. However a $`ℝ^{E}`$ can hold only *E* orthogonal directions (vectors) and there are a lot of semantic in a language (in large vocabulary).
<br/>
We would like the embedding space to hold relevat semantics as much as it can, however increasing E will result in space and computing cost. Nevertheless we can see that not so large embedding spaces supply the semantic's demand, and there is a hypothesis that tries to explain this phenomenon.<br/>
According to [*Johnson–Lindenstrauss lemma*](https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma) if we "cram" more vectors in the space and ease the rigid demand of [*Orthogonality*](https://en.wikipedia.org/wiki/Orthogonality) a little bit and allow a slight deviation, $`0<ε<1`$. Meaning we can arrange the vetors, not in exactly 90° between each other, but in a range of $`90°-ε \le ∡e_{i}e_{j} \le 90°+ε`$ between them, each vector will have an angle of $`[90°-ε , 90°+ε]`$ with all other vectors. Then the *lemma* tells us we can arrage D vecotrs in $`ℝ^{E}`$, when *D ≈* *****O*****$`\big( exp(E·ε^2) \big)`$.<br/>
For exapmle in $`ℝ^{100}`$ we can arrange ~exp(100) ≈ $`2.68·10^{43}`$ vectors/directions/semantics and that is a lot of semantics!


* Alternative method to embedding: *Token IDs* - token IDs id a simpe method which every token gets aunique integer. This is a more simple approch that reduce the computing and space complexity, However it misses the contextual connection between tokens because of that simplicity.
## Transformer
<img align="right" width="310"  src="https://github.com/user-attachments/assets/e55c4e75-3ed1-4b12-95d6-49bdf9dc10a6">

*The Transformer* is a deep learning architecture that was developed by researchers at Google and is based on the multi-head attention mechanism, which was proposed in the 2017 paper "Attention Is All You Need". The tranformer is Encoder-Decoder ... <br/>
### Attention
The [*Attention*](https://en.wikipedia.org/wiki/Attention_(machine_learning)) mechanism is the heart of the *Transformer* and it is a machine learning method that determines the relative importance of each component in a sequence relative to the other components in that sequence. In this method the learnable parameters are the weights: $`W_{Q}, W_{K}, W_{V}, W_{out}(optional)`$, we use them to create $`Q, K, V`$ and then execute the attention:

```math
Attention(Q,K,V) = Softmax \Bigg(\frac{Q K^{T}}{\sqrt{d}} \Bigg)·V
```
<br/>
When I mention 'attention' here I am speaking about 'Scaled Dot-Product Attention'.

#### Self-Attention
<img align="right" width="350"  src="https://github.com/user-attachments/assets/86b1234e-de87-4c88-bd53-e7c148769d2f">

Self-Attention is the simplest way of attention. we use our input and the weights to create the query matrix, *Q*, the key matrix, *K*, and the value matrix, *V*, and then execute the attention. this will tell us the affinity between vector(tokens/words). We can use it for various NLP tasks like text generation etc. 

Given an   $`Input:X∈ℝ^{M×E}`$, when `M=max_length` and `E=embedding_dimension`. 

$`X·W_{q} = Q∈ℝ^{M×d_k}`$  -  *query matrix*<br/>
$`X·W_{k} = K∈ℝ^{M×d_k}`$  -  *key matrix*<br/>
$`X·W_{v} = V∈ℝ^{M×d_v}`$  -  *value matrix*<br/>

$`d_{k}`$ - <br/>
$`d_{v}`$ - <br/><br/>

```math
Attention(Q,K,V) = Softmax \Bigg(\frac{Q K^{T}}{\sqrt{d}} \Bigg)·V = ΔX'∈ℝ^{M×d_v}
```

In case of: $`d_v \ne E`$ then we define a out matrix $`W_{out}∈ℝ^{d_v×E}`$. This matrix is also weigth matrix (has trained parameters) it can be used it to make the model more complex, and if not set $`d_v \ne E`$.<br/>
$`→ ΔX'·W_{out} = ΔX∈ℝ^{M×E}`$<br/>
$`⇨ Y = ΔX + X`$ - *Residual connection*<br/>


TODO: masked attention


#### Cross-Attention
<img align="right" width="330" height="400" src="https://github.com/user-attachments/assets/9f7bdae9-f051-42ae-973e-5dd9144fee09">

The difference between self attention and cross-attention...
input and conditional input, can be the target output or maybe prompt words

$`Input:  X∈ℝ^{M×N} ,  C∈ℝ^{L×E}`$

$`X·W_q = Q∈ℝ^{M×d_k}`$ - *query matrix*<br/>
$`C·W_k = K∈ℝ^{L×d_k}`$ - *key matrix*<br/>
$`C·W_v = V∈ℝ^{L×d_v}`$ - *value matrix*<br/>


```math
 = Softmax \Bigg(\frac{Q K^{T}}{\sqrt{d}} \Bigg)·V = ΔX'∈ℝ^{M×d_v}
```
$`→ ΔX'·W_{out} = ΔX∈ℝ^{M×E}`$<br/>
$`⇨ Y = ΔX + X`$ - *Residual connection*



### FeedForward

![feedforward](https://github.com/user-attachments/assets/484983aa-a374-4d71-bca1-f94467502650)



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
<br/>
**Example**:<br/>
Lets us note sequence length as M (M objects/tokens).

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

[3Blue 1Brown](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)


## Draft

```math 
Attention(Q,(K,V)) = \sum_{i=1}^M \alpha(q,k_{i})v_{i}
```
