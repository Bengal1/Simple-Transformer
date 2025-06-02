# Simple Transformer Guide

This is a practical guide for building [*Transformer*](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)), and it applies to intermediate ML programmers who like to know how to build a Transformer with Pytorch (if you are a beginner I will suggest [Simple CNN Guide](https://github.com/Bengal1/Simple-CNN-Guide)). <br/>
*The Transformer* is a deep learning architecture that was developed by researchers at Google and is based on the multi-head attention mechanism, which was proposed in the 2017 paper "Attention Is All You Need" <sup>[<a href="#ref1">1</a>]</sup>.
SimpleTransformer architecture is built according to article "Attention Is All You Need", In this project we will use it for [Machine Translation](https://en.wikipedia.org/wiki/Machine_translation).
This Repository is built for learning purposes, it contain theoretical and practical in knowledge and its goal is to help people who would like to start coding transformer executing [*NLP (Natural Language Processing)*](https://en.wikipedia.org/wiki/Natural_language_processing) tasks.

## Requirements
- [![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)](https://www.python.org/) <br/>
- [![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/) <br/>
- [![Datasets](https://img.shields.io/badge/HuggingFace-Datasets-FCC624?logo=huggingface&logoColor=black)](https://huggingface.co/datasets) <br/>
- [![spaCy](https://img.shields.io/badge/spaCy-NLP-09A3D5?logo=spacy&logoColor=white)](https://spacy.io/) <br/>

## Transformer
<img align="right" height="500"  src="https://github.com/user-attachments/assets/63544640-b22d-4c1e-94f3-d5c101ae05fd">

*The Transformer* is a deep learning architecture that was developed by researchers at Google and is based on the multi-head attention mechanism, which was proposed in the 2017 paper "Attention Is All You Need". The model is composed of an encoder and a decoder, each built from layers containing multi-head self-attention, feed-forward networks, residual connections, and layer normalization. The encoder processes the input sequence using self-attention and feed-forward layers to create context-aware representations. The decoder generates the output sequence using masked self-attention, attends to the encoder's output, and predicts the next token step by step. *Attention* is the core of the Transformer. It's what allows the model to weigh the importance of different words in a sequence—both in the input (via self-attention) and between input and output (via cross-attention). This mechanism replaces recurrence and convolution, making the model more efficient and better at capturing long-range dependencies. <br/>
This repository follow the original transformer from the paper with 6 encoder and 6 decoder layers and 8 heads for each multi-head attention, the rest can be noticed in the transformer architecture figure to th right, totaling `46,839,610` learnable parameters.

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
The Layers of the *FeedForward Network* consist of Dense layer, also called the fully-connected layer, and is used for abstract representations of input data. In this layer, neurons connect to every neuron in the preceding layer. In *Multilayer Perceptron* networks, these layers are stacked together. <br/> 
In our model the *Feed-Forward* network compose of 2 fully-connected layers and a ReLU activation that applied between them. I also applied *Dropout* according to "Attention Is All You Need". <br/> 
For a single Network 'layer', the output is calculated as:

```math
y = W_{2}·f(W_{1}·x+b_{1}) + b_{2}
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

`k` - Position of an object in the input sequence, $`0 \le k <M-1`$ (M=sequence length).<br/>
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

After calculating the positional encoding vectors, $`[p_0, p_1, p_2,..., p_{M-1}]`$, we add them to the embedding vectors, $`[e_0, e_1, e_2,..., e_{M-1}]`$ :<br/> 

$$
[e_0 + p_0,\hspace{0.3em} e_1 + p_1,\hspace{0.3em} e_2 + p_2,\hspace{0.2em}...,\hspace{0.2em} e_{M-1} + p_{M-1}]
$$


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

### Adam Optimizer
The Adam optimization algorithm<sup>[<a href="#ref2">2</a>]</sup> is an extension to stochastic gradient descent (SGD). Unlike SGD, The method computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients Adam combines the benefits of two other methods: momentum and RMSProp.

#### Adam Algorithm:
* $`\theta_t`$​ : parameters at time step *t*.
* $`\beta_1,\beta_2​`$: exponential decay rates for moments estimation.
* $`\alpha`$ : learning rate.
* $`\epsilon`$ : small constant to prevent division by zero.
* $`\lambda`$ : weight decay coefficient. <br/>

1. Compute gradients:

$$
g_t = \nabla_{\theta} J(\theta_t)
$$

2. Update first moment estimate (mean):

$$
m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t
$$

3. Update second moment estimate (uncentered variance):

$$
v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2
$$

4. Bias correction:

$$
\hat{v}_t = \frac{v_t}{1 - \beta_2^t} \quad ; \quad \hat{m}_t = \frac{m_t}{1 - \beta_1^t}
$$

5. Update parameters:

$$
\theta_{t+1} = \theta_t - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

* In our model *Weight decay* is applied:

$$
\theta_{t+1} = \theta_t - \alpha \cdot \Bigg( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \cdot \theta_t \Bigg)
$$

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

### Cross-Entropy Loss Function
This criterion computes the cross entropy loss between input logits and target. Loss function is a function that maps an event or values of one or more variables onto a real number intuitively representing some "loss" associated with the event. The Cross Enthropy Loss function is commonly used in classification tasks both in traditional ML and deep learning. It compares the predicted probability distribution over classes (logits) with the true class labels and penalizes incorrect or uncertain predictions.

$$
Loss = - \sum_{i=1}^{C} y_i \log(\hat{y}_i)
$$

Where:
* $`C`$  is the number of classes.
* $`y_i`$​  is the true probability for class *i* (usually 1 for the correct class and 0 for others).
* $`\hat{y}_i`$  is the predicted probability for class *i*.

### Teacher forcing
Teacher Forcing is a training strategy used in sequence-to-sequence (seq2seq) models, especially in tasks like machine translation, text generation, and speech recognition. During training, the model is fed the actual ground truth output from the previous time step instead of its own predicted output. This helps the model learn faster and improves convergence. During inference, it must generate each token from its own previous outputs, which can lead to cascading errors if one prediction is wrong — that's exposure bias. <br/>
Given an input: `This is Simple Transformer Guide!` and a target: `Ceci est un guide simple du Transformer!`, every iteration we will feed input from target:

<img src="https://github.com/user-attachments/assets/eb3cf86c-ea30-4f15-bf59-83a22261a3d7"  width="600"/>


Mistake at one step can lead to poor outputs later — exposure bias.

## Evaluation 

### Training & Validation Loss

`<img src="" align="center" width="1000"/>`

### Typical Run

`<img src="" align="center" width="1000"/>`

### Bilingual Evaluation Understudy (BLEU)

`<img src="" align="center" width="1000"/>`

## References
<b id="ref1">[1]</b> [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

<b id="ref2">[2]</b> [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)

