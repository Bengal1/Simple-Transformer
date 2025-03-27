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
        max_length (int): Maximum sequence length, including special tokens.
        pad_idx (int): Index of the padding token.
        unk_idx (int): Index of the unknown token.

    Class Attributes:
        en_vocab (dict): English vocabulary mapping tokens to indices. This dictionary is
                            updated dynamically with each dataset split.
        fr_vocab (dict): French vocabulary mapping tokens to indices. Like `en_vocab`, it expands
                            with each dataset instance.
        special_tokens (list): List of special tokens (`"<unk>"`, `"<pad>"`, `"<bos>"`, `"<eos>"`)
                            used for handling unknown words, padding, and marking sentence boundaries.
    """

    en_vocab = {"<unk>": 0, "<pad>": 1, "<bos>": 2, "<eos>": 3}
    fr_vocab = {"<unk>": 0, "<pad>": 1, "<bos>": 2, "<eos>": 3}
    special_tokens = ["<unk>", "<pad>", "<bos>", "<eos>"]

    def __init__(self, split="train", local_file=None):
        """Initializes the IWSLT14Dataset.

        Args:
            split (str, optional): The dataset split to load ('train', 'validation', or 'test'). Defaults to "train".
            local_file (str, optional): Path to a local dataset file. If provided, the dataset is loaded from this file
                                        instead of being fetched from Hugging Face.

        Raises:
            KeyError: If the specified split is not found in the local dataset file.
        """
        super().__init__()
        self.split = split  # Store split type
        self.local_file = local_file  # Debugging option

        self._load_tokenizers()
        self._load_dataset()
        self._update_global_vocab()
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

    def _update_global_vocab(self):
        """Updates the global English and French vocabularies with tokens from the current
        dataset split.
        """
        tokenized_en, tokenized_fr = self.tokenized_data[self.split]
        for token in set(token for sentence in tokenized_en for token in sentence):
            if token not in IWSLT14Dataset.en_vocab:
                IWSLT14Dataset.en_vocab[token] = len(IWSLT14Dataset.en_vocab)
        for token in set(token for sentence in tokenized_fr for token in sentence):
            if token not in IWSLT14Dataset.fr_vocab:
                IWSLT14Dataset.fr_vocab[token] = len(IWSLT14Dataset.fr_vocab)

    def _set_special_indices(self):
        """Sets indices for special tokens."""
        self.unk_idx = IWSLT14Dataset.en_vocab["<unk>"]
        self.pad_idx = IWSLT14Dataset.en_vocab["<pad>"]

    def _compute_max_length(self):
        """Determines the maximum sentence length in the dataset."""
        tokenized_en, tokenized_fr = self.tokenized_data[self.split]
        max_en_length = max(len(sentence) for sentence in tokenized_en)
        max_fr_length = max(len(sentence) for sentence in tokenized_fr)
        self.max_length = max(max_en_length, max_fr_length) + 2  # +2 for <bos> and <eos>

    def __len__(self) -> int:
        """Returns the number of sentence pairs in the selected split."""
        return len(self.tokenized_data[self.split][0])

    def __getitem__(self, idx: int) -> tuple:
        """Retrieves tokenized and padded sentences for the selected split.
        
        Args:
            idx (int): Index of the sentence pair.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Token indices for English and French sentences.
        """
        tokenized_en, tokenized_fr = self.tokenized_data[self.split]

        # Add <bos> and <eos> tokens
        en_sentence = ["<bos>"] + tokenized_en[idx] + ["<eos>"]
        fr_sentence = ["<bos>"] + tokenized_fr[idx] + ["<eos>"]

        # Pad sentences
        en_sentence += ["<pad>"] * (self.max_length - len(en_sentence))
        fr_sentence += ["<pad>"] * (self.max_length - len(fr_sentence))

        # Convert tokens to indices
        en_indices = [IWSLT14Dataset.en_vocab.get(token, self.unk_idx) for token in en_sentence]
        fr_indices = [IWSLT14Dataset.fr_vocab.get(token, self.unk_idx) for token in fr_sentence]

        return torch.tensor(en_indices), torch.tensor(fr_indices)

    @classmethod
    def get_vocab_sizes(cls) -> tuple:
        """Returns the total vocabulary size for both English and French."""
        return len(cls.en_vocab), len(cls.fr_vocab)

    def get_padding_index(self) -> int:
        """Returns the padding index for embedding layers."""
        return self.pad_idx

    def get_unknown_index(self) -> int:
        """Returns the unknown token index."""
        return self.unk_idx

    def get_max_length(self) -> int:
        """Returns the maximum sentence length, considering <bos> and <eos>."""
        return self.max_length
