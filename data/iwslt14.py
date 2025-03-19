import json
import torch
import spacy
from datasets import load_dataset
from torch.utils.data import Dataset


class IWSLT14Dataset(Dataset):
    """
    Custom PyTorch Dataset for the IWSLT14 English-French translation dataset.

    This dataset handles tokenization, vocabulary construction, and padding for training translation models.

    Attributes:
        split (str): The dataset split ('train', 'validation', or 'test').
        local_file (str, optional): Path to a local dataset file (if provided, loads from disk instead of Hugging Face).
        en_nlp (spacy.Language): spaCy tokenizer for English.
        fr_nlp (spacy.Language): spaCy tokenizer for French.
        en_vocab (dict): Token-to-index mapping for English.
        fr_vocab (dict): Token-to-index mapping for French.
        max_length (int): Maximum sequence length, including special tokens.
        pad_idx (int): Index of the padding token.
        unk_idx (int): Index of the unknown token.
    """

    def __init__(self, split="train", local_file=None):
        super().__init__()
        self.split = split  # Store split type
        self.local_file = local_file  # Debugging option
        self._load_tokenizers()
        self._load_dataset()
        self._build_vocabularies()
        self._set_special_indices()
        self._compute_max_length()

    def _load_tokenizers(self):
        """Loads spaCy tokenizers for English and French."""
        self.en_nlp = spacy.load("en_core_web_sm")
        self.fr_nlp = spacy.load("fr_core_news_sm")

    def _load_dataset(self):
        """Loads the specified split of the IWSLT14 dataset and tokenizes sentences.
        If a local file is provided, loads from disk. Otherwise, fetches data from Hugging Face.
        """
        if self.local_file:
            print(f"Loading local dataset from {self.local_file}...")
            with open(self.local_file, "r", encoding="utf-8") as f:
                iwslt_data = json.load(f)

            # Handle nested structure for splits like "train", "validation", "test"
            if self.split in iwslt_data:
                en_sentences = iwslt_data[self.split]["en"]
                fr_sentences = iwslt_data[self.split]["fr"]
            else:
                # This assumes the dataset does not contain the split key (i.e., it's flat)
                raise KeyError(f"The dataset for the split '{self.split}' is not found in the file.")

        else:
            # Load the dataset from Hugging Face
            print(f"Loading IWSLT14 {self.split} dataset...")
            iwslt_data = load_dataset("ahazeemi/iwslt14-en-fr", split=self.split)
            en_sentences = iwslt_data["en"]
            fr_sentences = iwslt_data["fr"]

        # Tokenize
        self.tokenized_data = {
            self.split: (
                [self._tokenize_text(sentence, self.en_nlp) for sentence in en_sentences],
                [self._tokenize_text(sentence, self.fr_nlp) for sentence in fr_sentences],
            )
        }

    @staticmethod
    def _tokenize_text(text: str, nlp_model) -> list:
        """Tokenizes a given text using the specified spaCy NLP model.

        Args:
            text (str): The input sentence.
            nlp_model: The spaCy NLP model to use.

        Returns:
            list: A list of tokenized words.
        """
        return [token.text for token in nlp_model(text)]

    def _build_vocabularies(self):
        """Builds English and French vocabularies based on tokenized data."""
        self.special_tokens = ["<unk>", "<pad>", "<bos>", "<eos>"]

        # Construct vocabulary from dataset tokens
        tokenized_en, tokenized_fr = self.tokenized_data[self.split]
        self.en_vocab = self._build_vocab(tokenized_en)
        self.fr_vocab = self._build_vocab(tokenized_fr)

    def _build_vocab(self, tokenized_sentences: list) -> dict:
        """Builds a vocabulary mapping from token to index.

        Args:
            tokenized_sentences (list): A list of tokenized sentences.

        Returns:
            dict: Token-to-index mapping.
        """
        unique_tokens = set(token for sentence in tokenized_sentences for token in sentence)
        return {token: idx for idx, token in enumerate(self.special_tokens + list(unique_tokens))}

    def _set_special_indices(self):
        """Sets indices for special tokens."""
        self.unk_idx = self.en_vocab["<unk>"]
        self.pad_idx = self.en_vocab["<pad>"]

    def _compute_max_length(self):
        """Determines the maximum sentence length in the dataset."""
        tokenized_en, tokenized_fr = self.tokenized_data[self.split]
        max_en_length = max(len(sentence) for sentence in tokenized_en)
        max_fr_length = max(len(sentence) for sentence in tokenized_fr)
        self.max_length = max(max_en_length, max_fr_length) + 2  # +2 for <bos> and <eos>

    def __len__(self) -> int:
        """Returns the number of sentence pairs in the selected split."""
        return len(self.tokenized_data[self.split][0])

    def __getitem__(self, idx: int):
        """Retrieves tokenized and padded sentences for the selected split."""
        tokenized_en, tokenized_fr = self.tokenized_data[self.split]

        # Add <bos> and <eos> tokens
        en_sentence = ["<bos>"] + tokenized_en[idx] + ["<eos>"]
        fr_sentence = ["<bos>"] + tokenized_fr[idx] + ["<eos>"]

        # Pad sentences
        en_sentence += ["<pad>"] * (self.max_length - len(en_sentence))
        fr_sentence += ["<pad>"] * (self.max_length - len(fr_sentence))

        # Convert tokens to indices
        en_indices = [self.en_vocab.get(token, self.unk_idx) for token in en_sentence]
        fr_indices = [self.fr_vocab.get(token, self.unk_idx) for token in fr_sentence]

        return torch.tensor(en_indices), torch.tensor(fr_indices)

    def get_padding_index(self) -> int:
        """Returns the padding index for embedding layers."""
        return self.pad_idx

    def get_unknown_index(self) -> int:
        """Returns the unknown token index."""
        return self.unk_idx

    def get_vocab_sizes(self):
        """Returns the sizes of the English and French vocabularies."""
        return len(self.en_vocab), len(self.fr_vocab)

    def get_max_length(self) -> int:
        """Returns the maximum sentence length, considering <bos> and <eos>."""
        return self.max_length
