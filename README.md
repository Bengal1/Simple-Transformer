TODO:
* BLEU score
* Teacher forcing
* token IDs
* number of parameters: 3,851,322
* scheduler learning rate
* NoamLR - ```lr=dmodel−0.5​×min(step−0.5,step×warmup_steps−1.5)```

# SimpleTransformer Guide

This is a practical guide for building [*Transformer*](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)), and it applies to beginners who like to know how to start building a Transformer with Pytorch. *The Transformer* is a deep learning architecture that was developed by researchers at Google and is based on the multi-head attention mechanism, which was proposed in the 2017 paper "Attention Is All You Need".<br/>
SimpleTransformer architecture is built according to article "Attention Is All You Need", In this project we will use it for [Machine Translation](https://en.wikipedia.org/wiki/Machine_translation). <br/>
This Repository is built for learning purposes, and its goal is to help people who would like to start coding transformer executing [*NLP (Natural Language Processing)*](https://en.wikipedia.org/wiki/Natural_language_processing) tasks.

## Requirements
* Python 3
* [Pytorch](https://pytorch.org/get-started/locally/)
* [Hugging Face](https://huggingface.co/)
* [spaCy](https://spacy.io/)

## Transformer
<img align="right" width="400"  src="https://github.com/user-attachments/assets/63544640-b22d-4c1e-94f3-d5c101ae05fd">

*The Transformer* is a deep learning architecture that was developed by researchers at Google and is based on the multi-head attention mechanism, which was proposed in the 2017 paper "Attention Is All You Need". The transformer is Encoder-Decoder ... <br/>
### Attention
The [*Attention*](https://en.wikipedia.org/wiki/Attention_(machine_learning)) (Scaled Dot-Product Attention) mechanism is the heart of the *Transformer* and, it is a machine learning method that determines the relative importance of each component in a sequence relative to the other components in that sequence. 
In this method we use the learnable (trainable) parameters are the weights: $`W_{Q}, W_{K}, W_{V}, W_{out}(optional)`$, create $`Q, K, V`$.

Given: &nbsp; $`W_{Q}∈ℝ^{E×d_k}`$ , &nbsp; $`W_{K}∈ℝ^{E×d_k}`$ , &nbsp; $`W_{V}∈ℝ^{E×d_v}`$  and Input &nbsp; $`X∈ℝ^{M×E}`$:

$$
X·W_{Q} = Q &ensp; ; &ensp; X·W_{K} = K &ensp; ; &ensp; X·W_{V} = V
$$

Each token in the input sequence is represented using three matrices: <br/>
***Query (Q)***: Represents the word we are currently processing and is used to find relevant words in the input. <br/>
***Key (K)***: Represents all words in the input sequence and is used to compare with the query to determine relevance. <br/>
***Value (V)***: Holds the actual word representations, which are combined based on attention scores to form the final output. <br/>

To determine which words are most relevant to the current query, we compute a dot product between *Q* and *K*, and in order to prevent extreme values, we scale the scores:  
```math
\frac{Q·K^{T}}{\sqrt{d}}
```
<br/>

To execute the attention we apply *Softmax* and multiply with *V* :

```math
Attention(Q,K,V) = Softmax \Bigg(\frac{Q K^{T}}{\sqrt{d}} \Bigg)·V
```
<br/>


### Self-Attention vs. Cross-Attention

*Self-Attention* is the simplest way of attention. we use the input sequence and the weights to create the query matrix, *Q*, the key matrix, *K*, and the value matrix, *V*, and then execute the attention. This will tell us the affinity between vectors(tokens/words). <br/>
In *Cross-Attention*, Q comes from the decoder's input (e.g., previously generated tokens or a prompt), while K and V come from the encoder's output, allowing the decoder to focus on relevant information from the input sequence. This means self-attention captures dependencies within a sequence, while cross-attention links information between two different sequences.
<br/>

Feature          | Self-Attention                                            | Cross-Attention
-----------------|-----------------------------------------------------------|------------------------------------------------------------------
Q (Query) Source | From the same sequence (input or decoder tokens)          | From the decoder’s conditional input (generated tokens or prompt)
K (Key) Source   | From the same sequence                                    | From the encoder’s output (context representations)
V (Value) Source | From the same sequence                                    | From the encoder’s output (context representations)
Purpose          | Captures dependencies within the same sequence            | Links information between encoder and decoder
Example          | Text summarization, sentiment analysis, language modeling | Machine translation, text-to-text generation, question answering


Given an ***Input***: $`X∈ℝ^{M×E}`$ and a ***Conditional Input***: $`C∈ℝ^{L×E}`$, when `M=max_length`, `E=embedding_dimension` and `L` is the conditional input sequence length. We compute the matrices of the attention: *Q*, *K* and *V*: <br/>

***Self-Attention:*** &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; ***Cross-Attention:*** <br/>
&emsp;$`X·W_{q} = Q∈ℝ^{M×d_k}`$ &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; $`X·W_q = Q∈ℝ^{M×d_k}`$ <br/>
&emsp;$`X·W_{k} = K∈ℝ^{M×d_k}`$ &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; $`C·W_k = K∈ℝ^{L×d_k}`$ <br/>
&emsp;$`X·W_{v} = V∈ℝ^{M×d_v}`$ &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; $`C·W_v = V∈ℝ^{L×d_v}`$ <br/>  

After computing the attention components, the rest of the process converges and is carried out in the same manner:

```math
Attention(Q,K,V) = Softmax \Bigg(\frac{Q K^{T}}{\sqrt{d}} \Bigg)·V = ΔX'∈ℝ^{M×d_v}
```
<br/>

In case of: $`d_v \ne E`$ then we use the out matrix $`W_{out}∈ℝ^{d_v×E}`$ to set the output in the right size. This matrix is also a weight matrix (has trainable parameters) it is also used to make the model more complex.<br/><br/>

$$
→ ΔX'·W_{out} = ΔX∈ℝ^{M×E}
$$
$$
 ⇨ Y = ΔX + X &ensp; (Residual - Connection)
$$

<br/>

* ***$`d_{k}`$  (Key dimension)***: The size of each key vector, which affects the scaling factor in the dot-product attention<br/>
* ***$`d_{v}`$  (Value dimension)***: The size of each value vector, determining the dimension of the weighted sum used as the attention output.<br/>
* ***Residual connection***: is a shortcut path that adds the input of the attention layer directly to its output before passing it to the next layer. This helps preserve the original input information, aids in gradient flow, and prevents vanishing gradients. In Transformers, the residual connection is followed by layer normalization to stabilize training.


### Masked-Attention
<img align="right" width="380" src="https://github.com/user-attachments/assets/b5d33ce5-2e29-4cb3-8da3-e572e716e447">

Masked attention is a variant of self-attention where certain positions in the attention matrix are masked (set to -∞ before softmax, Since $`e^{−∞}=0`$, so softmax turns the masked positions into zero attention) to prevent the model from attending to specific tokens. In Transformer decoders, causal masking is used to ensure that a token can only attend to previous tokens (not future ones), enabling autoregressive generation.
<br/><br/>

```math
Masked-Attention(Q,K,V) = Softmax \Bigg(\frac{Q K^{T}}{\sqrt{d}} + Mask \Bigg)·V
```
<br/>

### Multihead-Attention
<img align="right" width="220" src="https://github.com/user-attachments/assets/a0e99d43-2f85-4a85-980f-deba698aedfc">

Multi-head attention is an extension of the attention mechanism that allows the model to focus on different parts of the input sequence simultaneously, using multiple attention heads. Each head computes attention independently, and the results are combined to form a more comprehensive representation.<br/>
Instead of performing a single attention operation, multi-head attention runs multiple attention operations in parallel (with different parameterized projections) and then concatenates the results. Each head learns a different representation by attending to different parts of the input sequence. This allows the model to capture various kinds of dependencies in the input sequence simultaneously.

```math
head_i = Attention(QW_{Q_{i}},KW_{K_{i}},VW_{V_{i}})
```
```math
MultiHead-Attention = Concat(head_1,...,head_h)·W_{out}
```
<br/>

Where $`W_{Q_{i}}, W_{K_{i}}, W_{V_{i}}`$ and $`W_{out}`$ are learnable weight matrices.

For more information on Transformer and Attention there is a video series [3Blue 1Brown](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) 


--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### FeedForward Network
<img align="right" width="400"  src="https://github.com/user-attachments/assets/484983aa-a374-4d71-bca1-f94467502650">

A [*FeedForward Neural Network (FNN)*](https://en.wikipedia.org/wiki/Feedforward_neural_network) is a type of artificial neural network where connections between the nodes do not form cycles. The network consists of an input layer, one or more hidden layers, and an output layer. Information flows in one direction—from input to output—hence the name "feedforward."<br/>
The Layers of the *FeedForward Network* consist of Dense layer, also called the fully-connected layer, and is used for abstract representations of input data. In this layer, neurons connect to every neuron in the preceding layer. In *Multilayer Perceptron* networks, these layers are stacked together.<br/>
For a single layer, the output is calculated as:

```math
y = f(Wx+b)
```
Where:
* ***x*** is the input vector.
* ***W*** is the weight matrix.
* ***b*** is the bias vector.
* ***f*** is the activation function.

#### Activation Functions

The activation function introduces non-linearity into the network, allowing it to learn complex patterns.<br/> 
Common activation functions:

* ReLU (Rectified Linear Unit): $`f(x) = max(0,x)`$.
* Sigmoid: $`f(x) = {1 \over {1+e^{-x}}}`$.
* Tanh: $`f(x)=tanh(x)`$.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Positional Encoding
<img align="right" width="370"  src="https://github.com/user-attachments/assets/a69cfc70-d388-4411-be8a-86445603f879">

Positional encoding is a technique used in sequence-based models (such as transformers) to provide information about the positions or order of tokens in a sequence. Since transformers process entire sequences in parallel and lack an inherent mechanism for handling sequential order (unlike RNNs or LSTMs), positional encoding helps the model differentiate between tokens that appear in different positions within the sequence. Positional encodings are added to token embeddings, enabling the model to process both the semantic meaning and position of tokens in the sequence.<br/><br/>
  
$$
PE(k, 2i) = \sin\left( \frac{k}{n^{2i/d}} \right) \quad \text{;} \quad PE(k, 2i+1) = \cos\left( \frac{k}{n^{2i/d}} \right)
$$

<br/>

`k` - Position of an object in the input sequence, $`0 \le k <M`$ (M=sequence length).<br/>
`n` - User defined scalar. Set to 10,000 in the article "Attention Is All You Need".<br/>
`d` - Dimension of the model (output or output embedding space).<br/>
`i` - Used for mapping column's/object's indices,  $`0 \le i < \frac{2}{d}`$.<br/>
`PE(k,j)` - Positional encoding of thr j-th index in the k-th object in the input sequence.<br/>
<br/>
**Example**:<br/>
Lets us note sequence length as *M* (*M* objects/tokens).

$`PE(k=0) = [sin \Bigg(\frac{0}{10,000^{\frac{0}{d}}} \Bigg), cos \Bigg(\frac{0}{10,000^{\frac{0}{d}}} \Bigg), sin \Bigg(\frac{0}{10,000^{\frac{2}{d}}} \Bigg), cos \Bigg(\frac{0}{10,000^{\frac{2}{d}}} \Bigg),..., sin \Bigg(\frac{0}{10,000^{\frac{d-2}{d}}} \Bigg), cos \Bigg(\frac{0}{10,000^{\frac{d-2}{d}}} \Bigg)]`$<br/>
$`PE(k=1) = [sin \Bigg(\frac{1}{10,000^{\frac{0}{d}}} \Bigg), cos \Bigg(\frac{1}{10,000^{\frac{0}{d}}} \Bigg), sin \Bigg(\frac{1}{10,000^{\frac{2}{d}}} \Bigg), cos \Bigg(\frac{1}{10,000^{\frac{2}{d}}} \Bigg),..., sin \Bigg(\frac{1}{10,000^{\frac{d-2}{d}}} \Bigg), cos \Bigg(\frac{1}{10,000^{\frac{d-2}{d}}} \Bigg)]`$<br/>
.<br/>
.<br/>
.<br/>
$`PE(k=M-1) = [sin \Bigg(\frac{M-1}{10,000^{\frac{0}{d}}} \Bigg), cos \Bigg(\frac{M-1}{10,000^{\frac{0}{d}}} \Bigg), sin \Bigg(\frac{M-1}{10,000^{\frac{2}{d}}} \Bigg), cos \Bigg(\frac{M-1}{10,000^{\frac{2}{d}}} \Bigg),..., sin \Bigg(\frac{M-1}{10,000^{\frac{d-2}{d}}} \Bigg), cos \Bigg(\frac{M-1}{10,000^{\frac{d-2}{d}}} \Bigg)]`$<br/>

After calculating the positional encoding vectors, $`[p_1, p_2, p_3,..., p_M]`$, we add them to the embedding vectors, $`[e_1, e_2, e_3,..., e_M]`$ :<br/> $`[e_1 + p_1, e_2 + p_2, e_3 + p_3,..., e_M + p_M]`$


--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Normalization

*Normalization Layer* is used to stabilize and accelerate training by normalizing the inputs to each layer.<br/>
For each input vector (for each token in a sequence), subtract the mean and divide by the standard deviation of the vector's values. This centers the data around 0 with unit variance:
```math
x' = \frac{(x - μ)}{\sqrt{σ^{2} + ε}}
```
where *μ* is the mean and *σ* is the standard deviation of the input vector.<br/><br/>
Then apply scaling (gamma) and shifting (beta) parameters (trainable):

* *γ* (scale): A parameter to scale the normalized output.<br/>
* *β* (shift): A parameter to shift the normalized output.<br/>

```math
⇨  y = γ·x' + β
```

## Data
  
The IWSLT14 dataset is a multilingual parallel corpus created for machine translation tasks, specifically focusing on spoken language translation. It is part of the [*International Workshop on Spoken Language Translation (IWSLT)*](https://iwslt.org/) 2014 challenge. The dataset consists of TED Talks transcriptions and their translations, making it especially useful for training models that handle conversational and informal language.<br/>
The IWSLT14 English-French (En-Fr) dataset is a part of the International Workshop on Spoken Language Translation (IWSLT). The IWSLT14 dataset is specifically designed for *Machine Translation* tasks and contains parallel sentences in English and French. The dataset consists of sentence pairs aligned between English and French. Each sentence pair is a translation from one language to the other.<br/>
In this repository we load the dataset using Hugging Face's [*Dataset Library*](https://huggingface.co/datasets).

Dataset size:
* Training Set: Around 179,000 sentence pairs.
* Validation Set: About 903 sentence pairs.
* Test Set: Roughly 3,670 sentence pairs.
 
This dataset consists of 56K unique english tokens (vocabulary) and 73K unique french tokens. <br/>

### Tokenization
In order to prepare the data for training we need tokenization - convert words/sentences to tokens. The computer doesn't know what to do with words. when you feed it the sentence "This Simple Transformer Guide!" it doesn't understand the meaning of the words and the relations between them.<br/>
So what do computer understand? they understand numbers. in the core of computer it understands binary values ($`V_{low}`$ and $`V_{high}`$), but on higher levels it understand number and tensors (vectors, matrices, 3D matrices,...) and mathematical relation between them.
In order to provide the computer workable data we decompose the sentence into tokens and covert every token to a dense vector (process called *Embedding*).

```ruby
sentence = "This is Simple Transformer Guide!"
⇨ sentence_tokenized = ['This', 'is', 'Simple', 'Transformer', 'Guide', '!']
```
Before embedding, we would like to structure the data in such a way that it is easy for the transformer to receive it, so we will define a fixed length to sentences (input sequence) `max_length`, and then we pad sentence that are shorter (This is the method in use here).
* *Alternative method*: use max length 95% of the data. meaning 95% of the data will fit with no problem and 5% will be truncated according to size (the percentage can be changed, for example 90%). This approach allows you to handle the majority of the data, while avoiding excessively long sequences. Sacrificing 10% of data integrity to make the model smaller and more efficient.

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
sentence_tokenized = ['This', 'is', 'Simple', 'Transformer', 'Guide', '!']
⇨ sentence_for_embedding[max_length] = ['<bos>','This', 'is', 'Simple', 'Transformer', 'Guide', '!', '<eos>', '<pad>',..., '<pad>']
```

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Embedding
<img align="right" width="400"  src="https://github.com/user-attachments/assets/2cde7e51-70ed-4c5e-9575-33e8e0590083">

Embeddings are representations of values or objects like text, images, and audio that are designed to be consumed by machine learning models and semantic search algorithms. Embeddings translate objects like these into a mathematical form according to the preset factors, enable machine learning models to interact with various data types. <br/>
Word embedding is a technique used in Natural Language Processing (NLP) to represent words as dense numerical vectors. These vectors capture semantic relationships between words based on their context in large text corpora. <br/>
In our case we get a tokenized sequence (sentence, `M=max_length`) and we convert every token to a vector in the $`ℝ^{E}`$(`E=embedding_dim`, for more information on [*Real Vector Space*](https://en.wikipedia.org/wiki/Real_coordinate_space)) and we get for every sequence a matrix of size $`ℝ^{M×E}`$. <br/>


#### Intuitive understanding of Embedding

<img align="right" width="500"  src="https://github.com/user-attachments/assets/edf0a13e-fa50-4dbd-a040-940fcf3c0d76">

This explanation is for intuitive understanding of Embedding, you will need basic vector analysis to best understand it.<br/>
Lets assume we have the tokens `{'king', 'queen', 'man', 'woman'}` and we convert them to embedding vectors: $`\Big\{ e_{king}, e_{queen}, e_{man}, e_{woman} \Big\}`$, So for example we would expect, for good embedding, the next mathematical semantic connection:
```math
e_{king} - e_{queen} = e_{man} - e_{woman}
```

And we can interpret it as the gender difference between the vectors, meaning in the $`ℝ^{E}`$ embedding space (Lets assume E is big), there is a direction of gender, the more manly attributes the token has the further the vector will go in that direction and the same for womanly attributes in the opposite direction. 
We can also look at this mathematical semantic connection: 
```math
e_{king} - e_{man} = e_{queen} - e_{woman}
```
We can interpret it as if we strip the king from his gender then the vector that we get is the status/Royal vector as well as for the queen, meaning a royal direction.<br/>
And also it expected to get from the king vector to the queen vector we will do: 
&emsp;&emsp;&emsp;&emsp; $`e_{king} - e_{man} + e_{woman} = e_{queen}`$
<br/>
#### How Can $`ℝ^{E}`$ Holds Rich Language Semantics?

In Reality that is not what exactly happening. There is no equality in the mathematical connection, probably because there is more for king part to gender and royalty, but a rough axis direction can be noticed. We can interpret that for a some large vocabulary and $`ℝ^{E}`$, large embedding space, there will be semantic direction in this space. We expect them to be orthogonal, so that an object in this space when getting shifted in the 'Royal' direction it would not be shifted in unrelated direction like 'Size', 'Metallic', 'Temperature' and much more. Meaning larger the embedding space the more semantics it can hold. However, a $`ℝ^{E}`$ can hold only *E* orthogonal directions (vectors) and there are a lot of semantic in a language (in large vocabulary).
<br/>
We would like the embedding space to hold relevant semantics as much as it can, however increasing E will result in space and computing cost. Nevertheless, we can see that not so large embedding spaces supply the semantics demand, and there is a hypothesis that tries to explain this phenomenon.<br/>
According to [*Johnson–Lindenstrauss lemma*](https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma) if we "cram" more vectors in the space and ease the rigid demand of [*Orthogonality*](https://en.wikipedia.org/wiki/Orthogonality) a little bit and allow a slight deviation, $`0<ε<1`$. Meaning we can arrange the vectors, not in exactly 90° between each other, but in a range of $`90°-ε \le ∡e_{i}e_{j} \le 90°+ε`$ between them, each vector will have an angle of $`[90°-ε , 90°+ε]`$ with all other vectors. Then the *lemma* tells us we can arrange D vectors in $`ℝ^{E}`$, when *D ≈* *****O*****$`\big( exp(E·ε^2) \big)`$.<br/>
For example in $`ℝ^{100}`$ we can arrange ~exp(100·$`0.9^2`$) ≈ $`1.5·10^{35}`$ vectors/directions/semantics and that is a lot of semantics!

## Training and Optimization


## Loss & Typical Run 


## References
[Attention Is All You Need](https://arxiv.org/abs/1706.03762)

