# Simple Transformer
This repository is a guide and showcase for building a [*Transformer*](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)) with PyTorch. It is intended for intermediate ML practitioners who want to understand how to implement a Transformer from scratch (if you are a beginner, I recommend starting with the [Simple CNN Guide](https://github.com/Bengal1/Simple-CNN-Guide)).

The Transformer is a deep learning architecture introduced in the 2017 paper “Attention Is All You Need”<sup>[<a href="#ref1">1</a>]</sup>, based on the multi-head self-attention mechanism. [Large Language Models (LLMs)](https://en.wikipedia.org/wiki/Large_language_model), such as GPT, are direct descendants of this architecture. By scaling parameters, data, and compute, they have revolutionized NLP and enabled breakthroughs from machine translation to conversational AI.

In this project, I have implemented *SimpleTransformer* following the original paper’s design, and applied it to [Machine Translation](https://en.wikipedia.org/wiki/Machine_translation). This repository is built for learning purposes, combining theoretical background with practical implementation, to help those who want to start coding Transformers for [*NLP (Natural Language Processing)*](https://en.wikipedia.org/wiki/Natural_language_processing) tasks.

## Requirements
- [![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)](https://www.python.org/) <br/>
- [![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/) <br/>
- [![Datasets](https://img.shields.io/badge/HuggingFace-Datasets-FCC624?logo=huggingface&logoColor=black)](https://huggingface.co/datasets) <br/>
- [![spaCy](https://img.shields.io/badge/spaCy-NLP-09A3D5?logo=spacy&logoColor=white)](https://spacy.io/) <br/>
- [![sacrebleu](https://img.shields.io/pypi/v/sacrebleu?label=sacrebleu&color=blue&logo=python&logoColor=white&style=flat-square)](https://pypi.org/project/sacrebleu/) <br/>


## The Transformer

<img align="right" width="400" alt="Transformer_model_architecture" src="https://github.com/user-attachments/assets/248bb023-240a-443b-a971-c0f16c74fdfe" />

The Transformer is a deep learning architecture introduced by Google researchers in the 2017 paper “Attention Is All You Need.”
This paper not only proposed the Transformer architecture but also established the attention mechanism as a powerful alternative to recurrence and convolution for sequence modeling.
The model is built around multi-head attention, enabling it to efficiently capture complex relationships and long-range dependencies in sequences.

The Transformer consists of an encoder and a decoder, each composed of stacked layers that include multi-head attention, feed-forward networks, residual connections, and layer normalization.
The encoder encodes the input sequence into context-aware representations, while the decoder generates the output sequence step by step,
using masked self-attention to preserve autoregressive decoding and applying cross-attention over the encoder’s context representations.

The Transformer revolutionized sequence modeling by replacing recurrence and convolution with attention, allowing models to capture long-range dependencies more effectively and process sequences in parallel.
Its architecture enables rich, context-aware representations and dramatically improves performance across tasks like translation, summarization, and language understanding.
These capabilities set the stage for exploring its attention mechanism and core components, which are at the heart of its success.

### Attention
<img align="right" width="400" alt="Transformer_Encoder-Decoder" src="https://github.com/user-attachments/assets/1926cf27-ef25-465d-8c21-e3c9f6325d99" />

The [*Attention*](https://en.wikipedia.org/wiki/Attention_(machine_learning)) (Scaled Dot-Product Attention) mechanism is the heart of the *Transformer* and, it is a machine learning method that determines the relative importance of each component in a sequence relative to the other components in that sequence. 
In this method the learnable (trainable) parameters are the weights: $`W_{Q}, W_{K}, W_{V}, W_{out}(optional)`$, which creates $`Q, K, V`$.

Given: &nbsp; $`W_{Q}∈ℝ^{E×d_k}`$ , &nbsp; $`W_{K}∈ℝ^{E×d_k}`$ , &nbsp; $`W_{V}∈ℝ^{E×d_v}`$  and Input &nbsp; $`X∈ℝ^{M×E}`$:

$$
X·W_{Q} = Q &ensp; ; &ensp; X·W_{K} = K &ensp; ; &ensp; X·W_{V} = V
$$

When $E$ is the model/embedding dimension, $d_k$ is the dimension of the key and query vectors that control how similarities are computed, and $d_v$ is the dimension of the value vectors whose weighted sum forms the output.

Each token in the input sequence is represented using three matrices: <br/>
***Query (Q)***: Represents the word we are currently processing and is used to find relevant words in the input. <br/>
***Key (K)***: Represents all words in the input sequence and is used to compare with the query to determine relevance. <br/>
***Value (V)***: Holds the actual word representations, which are combined based on attention scores to form the final output. <br/>

To determine which words are most relevant to the current query, we compute a dot product between $Q$ and $K$, and in order to prevent extreme values we scale it, this is called the *Attention Score*:  
```math
\frac{Q·K^{T}}{\sqrt{d}}
```
<br/>

To execute the attention we apply $Softmax$ and multiply with $V$ and get the *Attention Weights*:

```math
Attention(Q,K,V) = Softmax \Bigg(\frac{Q K^{T}}{\sqrt{d}} \Bigg)·V
```
<br/>


### Self-Attention vs. Cross-Attention

*Self-Attention* is the simplest way of attention. we use the input sequence and the weights to create the query matrix, $Q$, the key matrix, $K$, and the value matrix, $V$, and then execute the attention. This will tell us the affinity between vectors(tokens/words). <br/>
In *Cross-Attention*, $Q$ comes from the decoder's input (e.g., previously generated tokens or a prompt), while $K$ and $V$ come from the encoder's output, allowing the decoder to focus on relevant information from the input sequence. This means self-attention captures dependencies within a sequence, while cross-attention links information between two different sequences.
<br/>


Feature          | Self-Attention                                            | Cross-Attention
-----------------|-----------------------------------------------------------|------------------------------------------------------------------
Q (Query) Source | From the same sequence (input or decoder tokens)          | From the decoder’s conditional input (generated tokens or prompt)
K (Key) Source   | From the same sequence                                    | From the encoder’s output (context representations)
V (Value) Source | From the same sequence                                    | From the encoder’s output (context representations)
Purpose          | Captures dependencies within the same sequence            | Links information between encoder and decoder
Example          | Text summarization, sentiment analysis, language modeling | Machine translation, text-to-text generation, question answering


Given an ***Input***: $`X∈ℝ^{M×E}`$ and a ***Conditional Input***: $`C∈ℝ^{L×E}`$, when $M$ is `max_length` of the input, $E$ is `embedding_dimension` and $L$ is the conditional input sequence length. We compute the matrices of the attention: $Q$, $K$ and $V$: <br/>

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
<img align="right" width="330" alt="masked self-attention" src="https://github.com/user-attachments/assets/51063c3b-7be3-4297-b6ba-7aded1303e31" />

Masked attention is a variant of self-attention where certain positions in the attention matrix are masked (set to -∞ before softmax, Since $`e^{−∞}=0`$, so softmax turns the masked positions into zero attention) to prevent the model from attending to specific tokens. In Transformer decoders, causal masking is used to ensure that a token can only attend to previous tokens (not future ones), enabling autoregressive generation.
<br/><br/>

```math
Masked-Attention(Q,K,V) = Softmax \Bigg(\frac{Q K^{T}}{\sqrt{d}} + Mask \Bigg)·V
```
<br/>

### Multi-Head Attention
<img align="right" width="230" alt="multihead_attention" src="https://github.com/user-attachments/assets/9b1e63f4-a200-4f94-9a02-5deccfea3b92" />

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


---

### Feed-Forward Network
<img align="right" width="420" alt="feedforward" src="https://github.com/user-attachments/assets/8e22c051-6c90-4ac2-b0ae-f46ccc59970f" />

A [*Feed-Forward Neural Network (FNN)*](https://en.wikipedia.org/wiki/Feedforward_neural_network) is a type of artificial neural network where connections between the nodes do not form cycles. The network consists of an input layer, one or more hidden layers, and an output layer. Information flows in one direction—from input to output—hence the name "feedforward."<br/>
The Layers of the *FeedForward Network* consist of Dense layer, also called the fully-connected layer, and is used for abstract representations of input data. In this layer, neurons connect to every neuron in the preceding layer. In *Multilayer Perceptron* networks, these layers are stacked together. <br/> 
In our model the *Feed-Forward* network compose of 2 fully-connected layers and a ReLU activation that applied between them. I also applied *Dropout* according to "Attention Is All You Need". <br/> 
For a single Network 'layer', the output is calculated as:

```math
y = f(W_{1}·x+b_{1})·W_{2} + b_{2}
```
Where:
* ***$`x`$*** is the input vector.
* ***$`W_i`$*** is the weight matrix of layer *i*.
* ***$`b_i`$*** is the bias vector of layer *i*.
* ***$`f`$*** is the activation function - ReLU.

The Dropout applies after the activation function.

#### Activation Functions

The activation function introduces non-linearity into the network, allowing it to learn complex patterns.<br/> 
Common activation functions:

* $ReLU$ (Rectified Linear Unit): $`f(x) = max(0,x)`$.
* $Sigmoid$: $`f(x) = {1 \over {1+e^{-x}}}`$.
* $Tanh$: $`f(x)=tanh(x)`$.
* $GELU$ (Gaussian Error Linear Unit): $`f(x) = x \cdot \frac{1}{2} \big[  1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right) \big]`$

---

### Positional Encoding
<img align="right" width="370"  src="https://github.com/user-attachments/assets/a69cfc70-d388-4411-be8a-86445603f879">

Positional encoding is a technique used in sequence-based models (such as transformers) to provide information about the positions or order of tokens in a sequence. Since transformers process entire sequences in parallel and lack an inherent mechanism for handling sequential order (unlike RNNs or LSTMs), positional encoding helps the model differentiate between tokens that appear in different positions within the sequence. Positional encodings are added to token embeddings, enabling the model to process both the semantic meaning and position of tokens in the sequence.<br/><br/>
  
$$
PE(k, 2i) = \sin\left( \frac{k}{n^{2i/d}} \right) \quad \text{;} \quad PE(k, 2i+1) = \cos\left( \frac{k}{n^{2i/d}} \right)
$$

<br/>

`k` - Position of an object in the input sequence, $`0 \le k <M-1`$ (M=sequence length).<br/>
`n` - User defined scalar. Set to 10,000 in the article "Attention Is All You Need".<br/>
`d` - Dimension of the model.<br/>
`i` - Used for mapping column's/object's indices,  $`0 \le i < \frac{2}{d}`$.<br/>
`PE(k,j)` - Positional encoding of thr j-th index in the k-th object in the input sequence.<br/>
<br/>
**Example**:<br/>
Lets us note sequence length as $M$ ($M$ objects/tokens). For sequence length $M$ and model dimension $d$, the positional encoding vector
for position $k$ is defined as:

$$
PE(k) = \Big[
\sin\Big(\tfrac{k}{10000^{0/d}}\Big),
\cos\Big(\tfrac{k}{10000^{0/d}}\Big),
\sin\Big(\tfrac{k}{10000^{2/d}}\Big),
\cos\Big(\tfrac{k}{10000^{2/d}}\Big),
\ldots,
\sin\Big(\tfrac{k}{10000^{(d-2)/d}}\Big),
\cos\Big(\tfrac{k}{10000^{(d-2)/d}}\Big)
\Big]
$$

Stacking the vectors $PE(k)$ for every position $k = 0, \dots, M-1$ forms the full positional encoding matrix used in the model:

$$
PE =
\begin{bmatrix}
\sin\frac{0}{10000^{0/d}} & \cos\frac{0}{10000^{0/d}} & \sin\frac{0}{10000^{2/d}} & \cos\frac{0}{10000^{2/d}} & \cdots & \sin\frac{0}{10000^{(d-2)/d}} & \cos\frac{0}{10000^{(d-2)/d}} \\
\sin\frac{1}{10000^{0/d}} & \cos\frac{1}{10000^{0/d}} & \sin\frac{1}{10000^{2/d}} & \cos\frac{1}{10000^{2/d}} & \cdots & \sin\frac{1}{10000^{(d-2)/d}} & \cos\frac{1}{10000^{(d-2)/d}} \\
\sin\frac{2}{10000^{0/d}} & \cos\frac{2}{10000^{0/d}} & \sin\frac{2}{10000^{2/d}} & \cos\frac{2}{10000^{2/d}} & \cdots & \sin\frac{2}{10000^{(d-2)/d}} & \cos\frac{2}{10000^{(d-2)/d}} \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
\sin\frac{M-1}{10000^{0/d}} & \cos\frac{M-1}{10000^{0/d}} & \sin\frac{M-1}{10000^{2/d}} & \cos\frac{M-1}{10000^{2/d}} & \cdots & \sin\frac{M-1}{10000^{(d-2)/d}} & \cos\frac{M-1}{10000^{(d-2)/d}}
\end{bmatrix}
$$

After calculating the positional encoding vectors, $`[p_0, p_1, p_2,..., p_{M-1}]`$, we add them to the embedding vectors, $`[e_0, e_1, e_2,..., e_{M-1}]`$ :<br/> 

$$
[e_0 + p_0,\hspace{0.3em} e_1 + p_1,\hspace{0.3em} e_2 + p_2,\hspace{0.2em}...,\hspace{0.2em} e_{M-1} + p_{M-1}]
$$


---

### Normalization
<img align="right" width="250"  src="https://github.com/user-attachments/assets/a1434118-a1d7-4a40-a35e-14b922ee0db4">

*Normalization Layer* is used to stabilize and accelerate training by normalizing the inputs to each layer.<br/>
For each input vector (for each token in a sequence), subtract the mean and divide by the standard deviation of the vector's values. This centers the data around 0 with unit variance:
```math
\hat{x} = \frac{(x - μ)}{\sqrt{σ^{2} + ε}}
```
where *μ* is the mean and *σ* is the standard deviation of the input vector.<br/><br/>
Then apply scaling (gamma) and shifting (beta) parameters (trainable):

* *γ* (scale): A parameter to scale the normalized output.<br/>
* *β* (shift): A parameter to shift the normalized output.<br/>

```math
⇨  y = γ·\hat{x} + β
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
In order to prepare the data for training, we need tokenization converting words or sentences into tokens. The computer doesn't know what to do with words. When you feed it the sentence "This Simple Transformer Guide!", it doesn't understand the meaning of the words or the relationships between them.<br/>
So what do computers understand? They understand numbers. At the core, computers deal with binary values (V<sub>low</sub> and V<sub>high</sub>), but at higher levels, they work with numbers and tensors (vectors, matrices, 3D tensors, and beyond) and the mathematical relationships between them. 
To make text workable, we first split the sentence into tokens and then map each token to a unique numerical ID from the vocabulary. These token IDs are what the model actually processes. 
Finally, these IDs are transformed into dense vectors through a process called embedding, which allows the model to learn semantic and syntactic relationships between tokens during training.

```ruby
sentence = "This is Simple Transformer Guide!"
⇨ tokenized_sentence = ['This', 'is', 'Simple', 'Transformer', 'Guide', '!']
⇨ sentence_of_tokenIDs = [73, 4, 871, 1082, 2374, 91]
```
Before embedding, we would like to structure the data in such a way that it is easy for the transformer to receive it, so we will define a fixed length to sentences (input sequence) `max_length`, and then we pad sentence that are shorter (This is the method in use here).
* *Alternative method*: use max length 95% of the data. meaning 95% of the data will fit with no problem and 5% will be truncated according to size (the percentage can be changed, for example 90%). This approach allows you to handle the majority of the data, while avoiding excessively long sequences. Sacrificing 10% of data integrity to make the model smaller and more efficient.

In order to give the model contextual sign and mange the data better, we use special tokens
```
special_tokens = ['<pad>', '<bos>', '<eos>', '<unk>']

<pad> - use for padding.
<bos> - beginning of sentence.
<eos> - end of sentence.
<unk> - unknown words.

We sets <unk> as the default. 
```
After sentence tokenization, we put before the sentence the beginning of sentence token, `<bos>`, and after it the end of sentence token, `<eos>`, and pad with padding token, `<pad>`, the remainder of the sentence up to `max_length`.<br/>
The unknown word token ,`<unk>`, use for words that are not in the vocabulary and dealing with failures, and for that reason we sets `<unk>` as the default. 
```ruby
sentence_tokenized = ['This', 'is', 'Simple', 'Transformer', 'Guide', '!']
⇨ sentence_for_embedding[max_length] = ['<bos>','This', 'is', 'Simple', 'Transformer', 'Guide', '!', '<eos>', '<pad>',..., '<pad>']
⇨ tokenIDs_for_embedding[max_length] = [1, 73, 4, 871, 1082, 2374, 91, 2, 0,..., 0]
```

---

### Embedding
<img align="right" width="400" alt="word_embed" src="https://github.com/user-attachments/assets/ff0b15d4-1092-414e-973c-aac3c4c2d70f" />

Embeddings are representations of values or objects like text, images, and audio that are designed to be consumed by machine learning models and semantic search algorithms. Embeddings translate objects like these into a mathematical form according to the preset factors, enable machine learning models to interact with various data types. <br/>
Word embedding<sup>[<a href="#ref3">3</a>]</sup> is a technique used in Natural Language Processing (NLP) to represent words as dense numerical vectors. These vectors capture semantic relationships between words based on their context in large text corpora. <br/>
In our case we get a tokenized sequence (sentence, `M=max_length`) and we convert every token to a vector in the $`ℝ^{E}`$(`E=embedding_dim`, for more information on [*Real Vector Space*](https://en.wikipedia.org/wiki/Real_coordinate_space)) and we get for every sequence a matrix of size $`ℝ^{M×E}`$. <br/>


#### Intuitive understanding of Embedding
<img align="right" width="500" alt="embed_space" src="https://github.com/user-attachments/assets/7808669b-756d-4ab6-a8a1-079cdea49ea8" />

This explanation is for intuitive understanding of Embedding. To best understand it, you will need a very basic vector analysis knowledge.<br/>
Lets assume we have the tokens `{'king', 'queen', 'man', 'woman'}` and we convert them to embedding vectors: $`\Big\{ e_{king}, e_{queen}, e_{man}, e_{woman} \Big\}`$. So for example we would expect a good embedding, the next mathematical semantic connection:

$$e_{king} - e_{queen} = e_{man} - e_{woman}$$

From the tokens 'king' and 'queen' we can assume royalty and gender of every token. from the tokens 'man' and 'woman' we can assume only gender. So if we subtract 'man' and 'woman' we get a gender difference vector, as well with 'king' and 'queen', because when we subtract them the will subtract the royalty direction (vector) of each other. 
We can interpret it as the gender difference between the vectors, meaning in the $`ℝ^{E}`$ embedding space (Lets assume E is big), there is a direction of gender, the more manly attributes the token has the further the vector will go in that direction and the same for womanly attributes in the opposite direction. 
We can also look at this mathematical semantic connection: 

$$e_{king} - e_{man} = e_{queen} - e_{woman}$$

We can interpret it as if we strip the king from his gender then the vector that we get is the status/Royal vector as well as for the queen, meaning a royal direction.<br/>
It also expected to get from the king vector to the queen vector we will use vector calculus and do: 

$$e_{king} - e_{man} + e_{woman} = e_{queen}$$

#### How Can $`ℝ^{E}`$ Holds Rich Language Semantics?

In Reality that is not what exactly happening. There is no equality in the mathematical connection, probably because there is more for king part to gender and royalty, but a rough axis direction can be noticed. 
We can interpret, that for a some large vocabulary and $`ℝ^{E}`$, large embedding space, there will be semantic direction in this space. We expect them to be orthogonal, so that an object in this space when getting shifted in the 'Royal' would do it with adding the 'Royal' vector (with scalar multiplication), and it would not be shifted in unrelated direction like 'Size', 'Metallic', 'Temperature' etc. <br/>
Meaning larger the embedding space the more semantics it can hold. However, a $`ℝ^{E}`$ can hold only *E* orthogonal directions (vectors) and there are a lot of semantic in a language (in a large vocabulary).
<br/>
We would like the embedding space to hold relevant semantics as much as it can, however increasing E will result in space and computing cost. Nevertheless, we can see that not so large embedding spaces supply the semantics demand, and there is a hypothesis that tries to explain this phenomenon.<br/>
According to [*Johnson–Lindenstrauss lemma*](https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma) if we "cram" more vectors in the space and ease the rigid demand of [*Orthogonality*](https://en.wikipedia.org/wiki/Orthogonality) a little bit and allow a slight deviation, $`0<ε<1`$. Meaning we can arrange the vectors, not in exactly 90° between each other, but in a range of $`90°-ε \le ∡e_{i}e_{j} \le 90°+ε`$ between them, each vector will have an angle of $`[90°-ε , 90°+ε]`$ with all other vectors. Then the *lemma* tells us we can arrange $D$ vectors in $`ℝ^{E}`$, when $D$ ≈ $`O\big( exp(E·ε^2) \big)`$.<br/>
For example in $`ℝ^{100}`$ we can arrange ~exp(100·$`0.9^2`$) ≈ $`1.5·10^{35}`$ vectors/directions/semantics and that is a lot of semantics!

## Training and Optimization

### Adam Optimizer
The Adam optimization algorithm<sup>[<a href="#ref2">2</a>]</sup> is an extension to stochastic gradient descent (SGD). Unlike SGD, The method computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients Adam combines the benefits of two other methods: momentum and RMSProp.

#### Adam Algorithm:
- $\theta_t$ : parameters at time step t.  
- $\beta_1, \beta_2$ : exponential decay rates for moment estimates.  
- $\alpha$ : learning rate.
- $\epsilon$ : small constant to prevent division by zero.  
- $\lambda$ : weight decay coefficient. <br/>


1. Compute gradients:
   <div align="center">
   $$g_t = \nabla_\theta J(\theta_t)$$
   </div>

2. Update moment estimates:
   <div align="center">
   $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad;\quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
   </div>
   
3. Bias correction: 
   <div align="center">
   $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t} \quad;\quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
   </div>
   
4. Parameter update: 
   <div align="center">
   $$\theta_{t+1} = \theta_t - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$
   </div>
   
* In our model *Weight decay* is applied:  
   <div align="center">
   $$\theta_{t+1} = \theta_t - \alpha \cdot \Bigg( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \cdot \theta_t \Bigg)$$
   </div>
   


### Noam Learning Rate
*NoamLR* scheduler was introduced in the original Transformer paper "Attention Is All You Need". Schedulers in deep learning are used to adjust the learning rate during training to improve convergence and performance. It sets the learning rate to increase linearly for a set number of warm-up steps and then decay proportionally to the inverse square root of the training step: 

$$
lr=d_{model}^{−0.5}​×\hspace{0.5em}min\bigg(Step^{−0.5},\hspace{0.1em}Step\hspace{0.3em}×\hspace{0.3em}Warmup^{−1.5}\bigg)
$$

Where:
* ***$`lr`$*** is the next learning rate.
* ***$`d_{model}`$*** is the model dimension (embedding dimension).
* ***$`Warmup`$*** is predefined hyperparameter.
* ***$`Step`$*** is optimizer current step.

This approach helps stabilize training in the early stages and allows the model to learn efficiently by avoiding large or unstable updates initially, while gradually reducing the learning rate to fine-tune the model later in training.

### Cross Entropy Loss Function
The Cross Entropy Loss Function is widely used for classification tasks, as it measures the difference between the predicted probability distribution and the true distribution. Given a predicted probability vector **$\hat{y}$** and a one-hot encoded target vector **$y$**, the loss for a single example is defined as:

$$
\mathcal{L}_{CE} = - \sum_{i} y_i \log \hat{y}_i
$$

This loss penalizes confident incorrect predictions more heavily than less certain ones, encouraging the model to assign higher probabilities to the correct classes. Minimizing cross-entropy effectively maximizes the likelihood of the correct labels under the model’s predicted distribution.

### Teacher forcing
<img align="right" width="500" src="https://github.com/user-attachments/assets/eb3cf86c-ea30-4f15-bf59-83a22261a3d7" />

Teacher Forcing<sup>[<a href="#ref4">4</a>]</sup> is a training strategy used in sequence-to-sequence (seq2seq) models, especially in tasks like machine translation, text generation, and speech recognition. During training, the model is fed the actual ground truth output from the previous time step instead of its own predicted output. This helps the model learn faster and improves convergence. During inference, it must generate each token from its own previous outputs, which can lead to cascading errors if one prediction is wrong — that's exposure bias. <br/>
Given an input: `This is Simple Transformer Guide!` and a target: `Ceci est un guide simple du Transformer!`, every iteration we will feed input from target:

Mistake at one step can lead to poor outputs later — exposure bias.

### Padding Mask
A padding mask in Transformers is a binary mask used to prevent the model from attending to <pad> tokens that were added to sequences to make them the same length in a batch. Without masking, the attention mechanism would treat these padding positions as valid input, potentially introducing meaningless information into the context. In practice, the padding mask has 0 (or False) where real tokens are and 1 (or True) where padding occurs, and it is applied before the softmax in the attention score computation by adding large negative values (`float('-inf')`) to the padded positions. This ensures the model focuses only on actual tokens when computing attention, improving both training stability and output quality.

### Beam Search
<img align="right" width="470" alt="beam_search_low_res" src="https://github.com/user-attachments/assets/0447e0fc-522c-41ca-bdc2-83fb6d997d51" />

Beam Search<sup>[<a href="#ref4">4</a>]</sup> is a decoding algorithm used to generate the most likely output sequence by keeping multiple hypotheses (beams) at each step, instead of just the best one (like greedy decoding). <br/>
This decoding strategy balances exploration and exploitation by keeping track of the top-k most likely partial sequences (beams) at each decoding step, rather than committing to the single most likely token as in greedy decoding. In greedy decoding, you pick the highest-probability token at each step, which is fast but can lead to suboptimal results because early mistakes cannot be corrected. Beam search instead expands all possible next tokens for each current beam, scores them (often using log-probabilities), and keeps only the best k sequences, allowing it to explore multiple promising paths in parallel. This often produces higher-quality translations or generations than greedy decoding, especially in Transformers, where the self-attention mechanism captures long-range dependencies that beam search can exploit to avoid short, repetitive, or incoherent outputs. However, beam search is slower than greedy decoding and can sometimes favor overly safe, generic sequences unless combined with techniques like length normalization or diverse beam search.

#### Length Normalization
Length normalization is a technique used in sequence generation (including Transformer decoding with beam search) to counteract the bias toward shorter outputs that arises when summing log-probabilities of tokens—since probabilities are less than 1, longer sequences naturally accumulate lower scores. Without normalization, the decoder might prefer prematurely ending the sequence with an <eos> token. Length normalization adjusts the total log-probability by dividing it by a function of the sequence length, like in this project, where the effective sequence length is first computed as the position of the first <eos> token (or the full length if <eos> is absent) and then passed through a penalty formula $`\Big(\frac{(5+L)}{6}\Big)^α`$, where *L* is the effective length and *α* controls the penalty strength. This ensures that beam search compares candidate sequences more fairly, allowing well-formed longer outputs to compete with shorter ones without being unfairly penalized by raw probability sums.<br/>

In beam search (or any sequence decoding), the log-probability of a candidate sequence is calculated by summing the log-probabilities of the tokens in that sequence:
```math
\log P(y_{1:T} \mid x) = \sum_{t=1}^T \log P(y_t \mid y_{1:t-1}, x)
```
where:
    $`y_{1:T}`$​ is the output sequence of length T.
    $`x`$ is the input (e.g., source sentence).
    $`P(y_{t}∣_{y1:t−1},x)`$ is the model’s predicted probability of token ytyt​ given the previous tokens and input.

In beam search with length normalization (like in your project), the score for each beam is:

```math
\text{score}(y_{1:T}) =
\frac{\sum_{t=1}^T \log P\left(y_t \mid y_{1:t-1}, x\right)}
{\text{length\_penalty}(T)}
```

where the $`\text{length\_penalty}(T)`$ is $`\Big(\frac{(5+L)}{6}\Big)^α`$, as described above.<br/>
This means beam search ranks sequences by their average (length-adjusted) log-probability rather than just raw probability sums, avoiding the bias toward short outputs.

The embedding layers are initialized with scaled normal distribution. Embedding normal distribution initialization means initializing embedding vectors by sampling each element from a normal (Gaussian) distribution with a small standard deviation (e.g., mean 0, std 0.01). This gives embeddings small random values before training begins, ensuring no initial bias toward any specific token. 

## Comparison with The Original Transformer
In my experiments, I focus on a single-model comparison using the IWSLT14 dataset, which contains approximately 180,000 sentence pairs, to evaluate how well the Transformer architecture performs under resource-constrained conditions. <be/>
For reference, the original Transformer models reported by Vaswani et al. (2017) were trained on the WMT 2014 dataset, which includes roughly 36 million sentence pairs, with a test set of about 3,000 sentences. In that setup, the Base model achieved a BLEU score of 38.1 and the Big model reached 41.0. While the paper also reports a Big Ensemble model achieving 41.8, ensembles are not within the scope of my comparison.<br/>
By focusing on a smaller dataset, I establish a fair baseline for translation quality under data-limited conditions, highlighting the impact of training scale rather than architectural differences.

My model differs from the original Transformer in several key aspects. I train a single-model Transformer on IWSLT14 using a single NVIDIA A100-SXM4-40GB GPU (Google Colab environment). I apply weight decay for regularization and adjust the batch size according to hardware limitations. To achieve a larger effective batch size without exceeding memory capacity, I use gradient accumulation, where gradients are accumulated over several smaller batches before performing an optimizer step. During inference, I rely on beam search with a length penalty to balance fluency and adequacy. These adjustments help stabilize training and improve generalization on the smaller dataset, while other architectural details such as the number of layers, embedding dimensions, attention heads, and weight initialization strategy remain consistent with the original paper.
<br/>

Model Variant                   | BLEU Score    | Dataset                          | Trainable Parameters
--------------------------------|---------------|----------------------------------|--------------------------
Original Transformer Base (512) |  38.1         |  WMT 2014 En-Fr (~36M samples)   | 65M parameters
Original Transformer Big (1024) |  41.0         |  WMT 2014 En-Fr (~36M samples)   | 213M parameters
Simple Transformer 512          |  35.35        |  IWSLT14 En-Fr (~180K samples)   | 147.8M parameters
Simple Transformer 1024         |  35.785       |  IWSLT14 En-Fr (~180K samples)   | 383.6M parameters

In comparing my model to both the Base and Big configurations of the original Transformer, I consider not only architectural scale (d_model, d_ff, num_heads) but also differences in parameter count and overall complexity. The Base model uses `d_model=512`, `d_ff=2048`, and `num_heads=8`, while the Big model doubles these values with `d_model=1024`, `d_ff=4096`, and `num_heads=16`. This scaling increases capacity and computational cost, but yields higher translation quality when sufficient data is available.

As shown in the table above, my implementation also differs in parameter count for two main reasons:
* **Weight tying**: The original Transformer shares parameters between source embeddings, target embeddings, and the output projection. I keep them separate, increasing the parameter count but allowing distinct representations for each component.
* **Vocabulary size**: The original uses a joint ~37K vocabulary, while I use separate vocabularies (~56K source, ~73K target), which increases the size of embedding and output layers but captures each language more precisely.

As a result, even when configured like Transformer-Base, my model contains more parameters, reflecting design choices that emphasize richer vocabulary coverage under data-limited conditions.

When evaluating our implementation against the original Transformer, it is important to note the stark difference in dataset size: the IWSLT14 corpus (~180K sentences) is only about 0.5% of the scale of WMT14 (~36M sentences). Despite this limitation, the Simple Transformer 512 achieved roughly 93% of the BLEU score of the original Base model, demonstrating that even with a much smaller dataset, a moderately sized model can generalize well. In contrast, the Simple Transformer 1024 reached only about 87% of the BLEU score of the original Big model. This suggests that while the larger model has substantially higher capacity, its complexity far exceeds the available data, leading to underutilization and diminished relative performance. In other words, the 512 variant strikes a better balance between model size and dataset scale, whereas the 1024 variant highlights the need for substantially more training data to fully leverage its capacity.

If you look at the training and validation curves presented in the following Evaluation section, they further illustrate these observations.. For the 512-dimensional model, both training and validation loss decrease steadily and stabilize without large divergence, while BLEU scores quickly plateau near their maximum, indicating efficient learning and good generalization. In contrast, the 1024-dimensional model shows a wider gap between training and validation loss, with validation loss plateauing at a higher value and BLEU scores improving more slowly. This reflects over-parameterization relative to the dataset size: the larger model can fit the training data better, but its higher capacity cannot be fully utilized given the limited training samples, leading to slower convergence and reduced generalization. Overall, these graphs confirm that the 512 variant achieves a better trade-off between complexity and dataset scale, while the 1024 variant would benefit from more data or stronger regularization.

## Evaluation 
The model performances are evaluated by two primary metrics *Loss* (training, validation & test) and *BLEU*.<br/>
In Optimizations problem, ML training, the Loss is the core signal guiding optimization. it measures how far model's predictions are from the targets, while optimization algorithms adjust model parameters to minimize it. Training loss is computed on the data used to update the model, reflecting how well it’s fitting that set, while validation loss is computed on unseen data to gauge generalization; a growing gap between them often signals overfitting.<br/>

BLEU (Bilingual Evaluation Understudy) is a metric for judging machine-generated text by comparing it to reference texts using n-gram overlaps. It combines these overlaps with a brevity penalty to avoid rewarding short outputs. Scores range from 0 to 1 (or 0–100%), with higher scores indicating closer matches, though it only measures exact wording matches.

### Evaluation on Test Dataset
Model Variant                   | Test Loss     | BLEU Score
--------------------------------|---------------|--------------
Simple Transformer 512          |  3.29         |  35.347
Simple Transformer 1024         |  3.377        |  35.785

### Training & Validation Loss
<img width="2560" height="1335" alt="loss_plot" src="https://github.com/user-attachments/assets/ec32215a-0afc-4801-8552-16334e1e7dd2" />

### Bilingual Evaluation Understudy (BLEU)
<img width="2560" height="1335" alt="bleu_plot" src="https://github.com/user-attachments/assets/6c7fc809-104b-4fd4-9270-8dbf04365461" />

## References
<b id="ref1">[1]</b> [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

<b id="ref2">[2]</b> [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)

<b id="ref3">[3]</b> [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781?utm_source=chatgpt.com)

<b id="ref4">[4]</b> [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215?utm_source=chatgpt.com)

