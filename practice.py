import spacy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from spacy.lang.fr import French

# Load spaCy models for tokenization
en_nlp = spacy.load("en_core_web_sm")
fr_nlp = spacy.load("fr_core_news_sm")

# Tokenizer function using spaCy
def tokenize_text(text, nlp_model):
    return [token.text for token in nlp_model(text)]

# Step 1: Load the dataset using Hugging Face `datasets` library
dataset = load_dataset("ahazeemi/iwslt14-en-fr")  # IWSLT14 English-French translation dataset

# Check the data structure
print(dataset)

# Tokenize the English and French sentences using spaCy
df = dataset['train']  # You can also use 'validation' or 'test' splits
tokenized_en = [tokenize_text(sentence, en_nlp) for sentence in df['en']]
tokenized_fr = [tokenize_text(sentence, fr_nlp) for sentence in df['fr']]

# Define Special Tokens
special_tokens = ["<unk>", "<pad>", "<bos>", "<eos>"]

# Build Vocabulary (simply based on the tokens in the dataset)
en_vocab = {token: idx for idx, token in enumerate(special_tokens + list(set([token for sublist in tokenized_en for token in sublist])))}
fr_vocab = {token: idx for idx, token in enumerate(special_tokens + list(set([token for sublist in tokenized_fr for token in sublist])))}

# Set Default Token for Unknown Words
unk_idx = en_vocab["<unk>"]
pad_idx = en_vocab["<pad>"]

# Define Maximum Sentence Length
max_length = max(max(len(x) for x in tokenized_en), max(len(x) for x in tokenized_fr)) + 2  # Add 2 for <bos> and <eos>

# Step 2: Dataset Class
class TranslationDataset(Dataset):
    def __init__(self, src_sentences, trg_sentences, max_len):
        self.src_sentences = src_sentences
        self.trg_sentences = trg_sentences
        self.max_len = max_len

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        # Add special tokens to the sentence
        src_sentence = ["<bos>"] + self.src_sentences[idx] + ["<eos>"]
        trg_sentence = ["<bos>"] + self.trg_sentences[idx] + ["<eos>"]

        # Pad the sentences if they are shorter than max_length
        src_sentence += ["<pad>"] * (self.max_len - len(src_sentence))
        trg_sentence += ["<pad>"] * (self.max_len - len(trg_sentence))

        return src_sentence, trg_sentence

# Step 3: Create Dataset & DataLoader
dataset = TranslationDataset(tokenized_en, tokenized_fr, max_length)
batch_size = 32
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Step 4: Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size_src, vocab_size_trg, embed_dim):
        super(TransformerModel, self).__init__()
        self.src_embedding = nn.Embedding(vocab_size_src, embed_dim, padding_idx=pad_idx)
        self.trg_embedding = nn.Embedding(vocab_size_trg, embed_dim, padding_idx=pad_idx)

    def forward(self, src_tokens, trg_tokens, src_vocab, trg_vocab):
        # Convert tokens to indices
        src_indices = torch.tensor([[src_vocab.get(token, unk_idx) for token in sentence] for sentence in src_tokens])
        trg_indices = torch.tensor([[trg_vocab.get(token, unk_idx) for token in sentence] for sentence in trg_tokens])

        # Convert token IDs to embeddings
        src_embedded = self.src_embedding(src_indices)
        trg_embedded = self.trg_embedding(trg_indices)

        return src_embedded, trg_embedded

# Step 5: Initialize Model
embed_dim = 256
model = TransformerModel(len(en_vocab), len(fr_vocab), embed_dim)

# Example Usage
for src_tokens, trg_tokens in data_loader:
    src_emb, trg_emb = model(src_tokens, trg_tokens, en_vocab, fr_vocab)
    print("Source embeddings shape:", src_emb.shape)
    print("Target embeddings shape:", trg_emb.shape)
    break
