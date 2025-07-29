"""
Handles loading and preprocessing the IWSLT14 English-French dataset.

This module defines the `IWSLT14Dataset` class, which supports loading the
IWSLT14 dataset from Hugging Face or a local file, tokenizing English and
French sentences using spaCy, constructing vocabularies, and padding sequences
based on a configurable percentile of input lengths.

Classes:
    IWSLT14Dataset: Manages dataset loading, tokenization, vocabulary building,
        and returns PyTorch-compatible dataset splits.
    _IWSLT14SplitDatasetView: A lightweight wrapper around tokenized splits
        for integration with PyTorch's DataLoader.

Features:
    - Loads English-French translation pairs from IWSLT14.
    - Tokenizes using spaCy with optional language model downloads.
    - Builds separate vocabularies for English (source) and French (target)
      from training data only.
    - Adds <pad>, <bos>, <eos>, <unk> special tokens to each vocabulary.
    - Pads sequences to a length based on a specified percentile (default: 95).
    - Returns split views suitable for use in PyTorch training loops.
"""


import json
import torch
import spacy
import math
import logging
from datasets import load_dataset
from typing import Optional

class IWSLT14Dataset:
    """
    A dataset class for loading and preprocessing the IWSLT14 English-French translation dataset.

    Handles loading from Hugging Face or local files, tokenization using spaCy,
    vocabulary construction, and sequence padding for training sequence-to-sequence
    models.

    Attributes:
        SPECIAL_TOKENS_CONFIG (dict): Configuration for special token strings and IDs.
        local_files (dict): Optional mapping of split names to local file paths.
        max_length (int): Maximum sequence length (auto-computed unless overridden).
        en_vocabulary (dict): English (source) token-to-ID mapping.
        fr_vocabulary (dict): French (target) token-to-ID mapping.
        tokenized_datasets (dict): Tokenized sentence data for all splits.
        en_nlp (spacy.Language): spaCy English tokenizer.
        fr_nlp (spacy.Language): spaCy French tokenizer.
    """

    SPECIAL_TOKENS_CONFIG = {
        "<pad>": {"default_id": 0, "attr_name": "pad_idx"},
        "<bos>": {"default_id": 1, "attr_name": "bos_idx"},
        "<eos>": {"default_id": 2, "attr_name": "eos_idx"},
        "<unk>": {"default_id": 3, "attr_name": "unk_idx"},
    }

    def __init__(self,
                 local_files: dict[str, str] = None,
                 max_length: int = None):
        """Initializes the dataset and triggers loading, tokenization, and
           vocabulary construction.

        Args:
            local_files (dict, optional): Local file paths for train/validation/test splits.
            max_length (int): Max length for padded sequences (used only as fallback).
        """
        self.local_files = local_files if local_files is not None else {}
        self.max_length = max_length

        self.en_vocabulary = {}
        self.fr_vocabulary = {}

        # Initialize vocabularies with special tokens
        logging.info("Initializing vocabularies with special tokens "
                     "based on SPECIAL_TOKENS_CONFIG.")
        for token_str, config in self.SPECIAL_TOKENS_CONFIG.items():
            self.en_vocabulary[token_str] = config["default_id"]
            self.fr_vocabulary[token_str] = config["default_id"]

        self.tokenized_datasets = {}

        # Loads spaCy tokenizers for English and French.
        logging.info("Loading spaCy English and French models.")
        # self.en_nlp = spacy.load("en_core_web_sm")
        # self.fr_nlp = spacy.load("fr_core_news_sm")
        try:
            self.en_nlp = spacy.load("en_core_web_sm")
            logging.info("SpaCy English model 'en_core_web_sm' loaded successfully.")
        except OSError:
            logging.error("SpaCy English model 'en_core_web_sm' not found. "
                          "Please run: python -m spacy download en_core_web_sm")
            raise

        try:
            self.fr_nlp = spacy.load("fr_core_news_sm")
            logging.info("SpaCy French model 'fr_core_news_sm' loaded successfully.")
        except OSError:
            logging.error("SpaCy French model 'fr_core_news_sm' not found. "
                          "Please run: python -m spacy download fr_core_news_sm")
            raise

        self._load_all_dataset()
        logging.info("All dataset splits loaded and tokenized.")

        self._update_vocabularies()
        logging.info("Vocabularies built and special indices set.")

        self._compute_max_length()
        logging.debug(f"Maximum sequence length computed: {self.max_length}")

        print("Finished loading all dataset splits and building vocabulary.\n")
        # logging.info("Finished loading all dataset splits and building vocabulary.")

    def _load_all_dataset(self):
        """Loads and tokenizes all splits (train, validation, test).

        Uses either local JSON files or downloads from Hugging Face. Tokenized data
        is stored internally for later vocabulary building and dataset creation.
        """
        for split_name in ["train", "validation", "test"]:
            if split_name not in self.tokenized_datasets:
                self.tokenized_datasets[split_name] = {"en": [], "fr": []}

        for split_name, file_path in self.local_files.items():
            if file_path:  # If a local file path is provided for the split
                with open(file_path, "r", encoding="utf-8") as f:
                    iwslt_data = json.load(f)

                if split_name in iwslt_data:
                    en_sentences = iwslt_data[split_name]["en"]
                    fr_sentences = iwslt_data[split_name]["fr"]
                else:
                    raise KeyError(
                        f"The dataset for the split '{split_name}' is not found "
                        f"in the file '{file_path}'.")

                # Tokenize and store directly into self.tokenized_datasets
                self.tokenized_datasets[split_name]["en"] = [
                    self._tokenize_text(sentence, self.en_nlp) for sentence in
                    en_sentences]
                self.tokenized_datasets[split_name]["fr"] = [
                    self._tokenize_text(sentence, self.fr_nlp) for sentence in
                    fr_sentences]

            else:  # Load the dataset from Hugging Face for the specific split_name
                print(f"Loading IWSLT14 {split_name} dataset from Hugging Face...")
                try:
                    iwslt_data = load_dataset("ahazeemi/iwslt14-en-fr",
                                              split=split_name)
                    en_sentences = iwslt_data["en"]
                    fr_sentences = iwslt_data["fr"]

                    # Tokenize and store directly into self.tokenized_datasets
                    self.tokenized_datasets[split_name]["en"] = [
                        self._tokenize_text(sentence, self.en_nlp) for sentence in
                        en_sentences]
                    self.tokenized_datasets[split_name]["fr"] = [
                        self._tokenize_text(sentence, self.fr_nlp) for sentence in
                        fr_sentences]
                except Exception as e:
                    print(
                        f"Could not load split '{split_name}' from Hugging Face: {e}")

    @staticmethod
    def _tokenize_text(text: str,
                       nlp_model: spacy.Language) -> list[str]:
        """Tokenizes a single sentence using the provided spaCy NLP model.

        Args:
            text (str): The input sentence.
            nlp_model (spacy.Language): A loaded spaCy tokenizer.

        Returns:
            list[str]: A list of tokenized words.
        """
        return [token.text for token in nlp_model(text)]

    def _update_vocabularies(self):
        """Builds English and French vocabularies from the training split.

        This method processes the tokenized training data to extract all unique tokens
        in both source (English) and target (French) languages. Assigns unique
        integer IDs to each token, and stores the resulting mappings.

        Additionally, assign special token IDs to class variables and estimates an
        appropriate maximum sequence length for padding and truncation based on the
        distribution of sentence lengths in the training set.
        """
        if "train" not in self.tokenized_datasets:
            logging.warning("'train' split not found in self.tokenized_datasets. "
                            "Vocabulary will be empty.")
            return

        # Update English vocabulary using only the 'train' split
        self._process_vocabulary_for_language("en", ["train"])

        # Update French vocabulary using only the 'train' split
        self._process_vocabulary_for_language("fr", ["train"])

        # Set special tokens variables
        self. _set_special_indices()

        # Set maximum sentence length
        self._compute_max_length()

    def _process_vocabulary_for_language(self,
                                         lang_code: str,
                                         splits_to_consider: list):
        """Adds all unique tokens from specified splits into the appropriate vocabulary.

        Args:
            lang_code (str): Language code ('en' or 'fr').
            splits_to_consider (list): List of split names to extract tokens from.

        Raises:
            ValueError: If an unsupported language code is given.
        """
        unique_tokens_across_specified_splits = set()
        for split_name in splits_to_consider:
            if (split_name in self.tokenized_datasets and lang_code in
                    self.tokenized_datasets[split_name]):
                for sentence_tokens in self.tokenized_datasets[split_name][lang_code]:
                    unique_tokens_across_specified_splits.update(sentence_tokens)

        if lang_code == "en":
            target_vocab = self.en_vocabulary
        elif lang_code == "fr":
            target_vocab = self.fr_vocabulary
        else:
            raise ValueError(
                f"Unsupported language code '{lang_code}'. Expected 'en' or 'fr'.")
        self.add_tokens_to_vocabulary(list(unique_tokens_across_specified_splits),
                                      lang_code)

    def _set_special_indices(self):
        """Assigns instance attributes for special token indices.

        Verifies presence and correctness of special tokens in the vocabulary.

        Raises:
            ValueError: If any special token is missing or incorrectly indexed.
        """
        logging.info("Setting special token indices.")

        for token_str, config in self.SPECIAL_TOKENS_CONFIG.items():
            attr_name = config["attr_name"]
            expected_id = config["default_id"]

            if token_str in self.en_vocabulary and token_str in self.en_vocabulary:
                actual_id_en = self.en_vocabulary[token_str]
                actual_id_fr = self.fr_vocabulary[token_str]
                if actual_id_en == expected_id and actual_id_fr == expected_id:
                    # Token found and has the CORRECT ID
                    setattr(self, attr_name, expected_id)
                    logging.debug(f"Set '{attr_name}' for token '{token_str}' to "
                                  f"correct index {expected_id}.")
                else:
                    # Critical Error: Token found, but its ID is WRONG
                    error_details = []
                    if actual_id_en != expected_id:
                        error_details.append(
                            f"English: {actual_id_en} (expected {expected_id})")
                    if actual_id_fr != expected_id:
                        error_details.append(
                            f"French: {actual_id_fr} (expected {expected_id})")

                    error_msg = (
                        f"Critical Error: Special token '{token_str}' found in "
                        f"vocabulary but has incorrect ID(s):"
                        f" {'; '.join(error_details)}. Its designated position was "
                        f"likely taken by another token. This is an unrecoverable "
                        f"vocabulary setup error."
                    )
                    logging.critical(error_msg)
                    raise ValueError(error_msg)
            else:
                # Critical Error: Special token is completely missing from vocabulary
                error_msg = (
                    f"Critical Error: Special token '{token_str}' not found in the "
                    f"English vocabulary. This token is essential for proper model "
                    f"operation and should have ID {expected_id}. Ensure the "
                    f"vocabulary building process correctly includes all defined "
                    f"SPECIAL_TOKENS_CONFIG entries. This is an unrecoverable setup "
                    f"error."
                )
                logging.critical(error_msg)
                raise ValueError(error_msg)

        logging.info("Special token indices verification and setup complete.")

    def _compute_max_length(self, percentile: float = 0.95):
        """Computes the padded sequence length using a percentile of training data lengths.

        Args:
            percentile (float): Percentile of sentence lengths to use (0 < p <= 1.0).

        Raises:
            ValueError: If the percentile is outside the valid range.
        """
        if self.max_length is not None:
            return

        if "train" not in self.tokenized_datasets or \
                   not self.tokenized_datasets["train"].get("en") or \
                   not self.tokenized_datasets["train"].get("fr"):
            logging.warning("'train' split not found or its language data is empty. "
                            "Cannot compute max_length.")
            self.max_length = 50  # Set a default fallback length
            return

        all_en_lengths = [len(s) for s in self.tokenized_datasets["train"]["en"]]
        all_fr_lengths = [len(s) for s in self.tokenized_datasets["train"]["fr"]]

        all_training_lengths = all_en_lengths + all_fr_lengths

        if not all_training_lengths:
            logging.warning(
                "Combined training data is empty. Cannot compute max_length.")
            self.max_length = 50  # Set a default fallback length
            return

        all_training_lengths.sort()

        if percentile == 1.0:
            computed_max_len = all_training_lengths[-1]
        elif 0 < percentile < 1.0:
            index = math.ceil(percentile * len(all_training_lengths)) - 1
            index = max(0, index)  # Ensure index is not negative
            computed_max_len = all_training_lengths[index]
        else:
            raise ValueError("Percentile must be between 0 and 1.0.")

        # Add 2 for <bos> and <eos> tokens
        self.max_length = computed_max_len + 2

        logging.info(f"Computed max_length (at {percentile * 100}% percentile) for "
                     f"training data: {self.max_length}")

    def add_tokens_to_vocabulary(self, tokens: list[str], target_language: str):
        """
                Adds new tokens to the specified language's vocabulary.

                Args:
                    tokens (List[str]): A list of token strings to add.
                    target_language (str): The language code ('en' for English,
                                'fr' for French) to which the tokens should be added.

                Raises:
                    ValueError: If an unsupported language code is provided.
                """
        if target_language == "en":
            target_vocab = self.en_vocabulary
        elif target_language == "fr":
            target_vocab = self.fr_vocabulary
        else:
            raise ValueError(f"Unsupported language code '{target_language}'. "
                             f"Expected 'en' or 'fr'.")

        for token in tokens:
            if token not in target_vocab:
                target_vocab[token] = len(target_vocab)
        logging.info(f"Added {len(tokens)} tokens to the {target_language} "
                     f"vocabulary.")

    def get_vocabularies(self) -> tuple[dict, dict]:
        """
        Returns the English and French vocabularies.

        Raises:
            RuntimeError: If vocabularies are empty (indicating a failure in population).

        Returns:
            tuple[dict, dict]: A tuple containing the English and French vocabularies.
        """
        if not self.en_vocabulary:
            error_msg = ("English vocabulary is empty. Ensure vocabulary building "
                         "completed successfully.")
            logging.critical(error_msg)
            raise RuntimeError(error_msg)

        if not self.fr_vocabulary:
            error_msg = ("French vocabulary is empty. Ensure vocabulary building "
                         "completed successfully.")
            logging.critical(error_msg)
            raise RuntimeError(error_msg)

        logging.debug("Returning initialized vocabularies.")
        return self.en_vocabulary, self.fr_vocabulary

    @classmethod
    def get_special_tokens_dict(cls) -> dict:
        """Returns a dictionary of special tokens and their default IDs.

        Returns:
            dict: Mapping from token strings to IDs.
        """
        return {token_str: config["default_id"]
                for token_str, config in cls.SPECIAL_TOKENS_CONFIG.items()}

    @classmethod
    def get_special_tokens_list(cls) -> list:
        """Returns a list of special token strings.

        Returns:
            list: List of special tokens.
        """
        return list(cls.SPECIAL_TOKENS_CONFIG.keys())

    def get_padding_index(self) -> int:
        """Returns the index used for padding tokens.

        Returns:
            int: Padding token index.
        """
        return self.pad_idx

    def get_vocabularies_sizes(self) -> tuple[int, int]:
        """Returns the size of the English and French vocabularies.

        Returns:
            tuple[int, int]: (size of en_vocab, size of fr_vocab)
        """
        return len(self.en_vocabulary), len(self.fr_vocabulary)

    def get_max_length(self):
        """Returns the maximum sequence length used for padding.

        Returns:
            int: Max sequence length.
        """
        return self.max_length

    def get_split_dataset(self, split_name: str):
        """Returns a dataset view for a specific split (train/validation/test).

        Args:
            split_name (str): Name of the split to retrieve.

        Returns:
            _IWSLT14SplitDatasetView: Dataset wrapper for the specified split.

        Raises:
            ValueError: If the split is not found.
        """
        if split_name not in self.tokenized_datasets:
            raise ValueError(f"Split '{split_name}' not available in the dataset.")

        return _IWSLT14SplitDatasetView(self.tokenized_datasets[split_name],
                                       self.en_vocabulary,
                                       self.fr_vocabulary,
                                       self.pad_idx,
                                       self.bos_idx,
                                       self.eos_idx,
                                       self.unk_idx,
                                       self.max_length)

    def get_datasets(self) -> tuple:
        """Returns dataset views for all splits (train, validation, test).

        Returns:
            tuple: (train_dataset, validation_dataset, test_dataset)
        """
        train_dataset      = self.get_split_dataset("train")
        validation_dataset = self.get_split_dataset("validation")
        test_dataset       = self.get_split_dataset("test")

        return train_dataset, validation_dataset, test_dataset


class _IWSLT14SplitDatasetView(torch.utils.data.Dataset):
    """
    PyTorch-compatible Dataset class for a single IWSLT14 split.

    Used to index and retrieve padded input/output token sequences.

    Attributes:
        split_data (dict): Dictionary with 'en' and 'fr' tokenized data.
        en_vocab (dict): Source language vocabulary.
        fr_vocab (dict): Target language vocabulary.
        pad_idx (int): Padding token index.
        bos_idx (int): Beginning-of-sentence token index.
        eos_idx (int): End-of-sentence token index.
        unk_idx (int): Unknown token index.
        max_length (int): Maximum sequence length to pad/truncate.
    """

    def __init__(self,
                 split_data: dict,
                 en_vocab: dict[str, int],
                 fr_vocab: dict[str, int],
                 pad_idx: int, bos_idx: int, eos_idx: int, unk_idx: int,
                 max_length: int):
        super().__init__()
        self.tokenized_data = split_data
        self.en_vocabulary = en_vocab
        self.fr_vocabulary = fr_vocab
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.unk_idx = unk_idx
        self.max_length = max_length


    def __len__(self) -> int:
        """Returns the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.tokenized_data["en"])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Fetches and process the source and target sequences at a given index.

        Add '<bos>' and '<eos>' to every sentence and padded it from '<eos>' up
        to max_length

        Args:
            idx (int): Index of the example to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing padded source and
                                                target tensors.
        """
        en_sentence = ["<bos>"] + self.tokenized_data["en"][idx] + ["<eos>"]
        fr_sentence = ["<bos>"] + self.tokenized_data["fr"][idx] + ["<eos>"]

        # Truncate if sentence is longer than max_length
        if len(en_sentence) > self.max_length:
            en_sentence = en_sentence[:self.max_length - 1] + ["<eos>"] # Ensure EOS
        else:
            en_sentence += ["<pad>"] * (self.max_length - len(en_sentence))

        if len(fr_sentence) > self.max_length:
            fr_sentence = fr_sentence[:self.max_length - 1] + ["<eos>"]
        else:
            fr_sentence += ["<pad>"] * (self.max_length - len(fr_sentence))

        en_indices = [self.en_vocabulary.get(token,
                                             self.unk_idx) for token in en_sentence]
        fr_indices = [self.fr_vocabulary.get(token,
                                             self.unk_idx) for token in fr_sentence]

        return (torch.tensor(en_indices, dtype=torch.long),
                torch.tensor(fr_indices, dtype=torch.long))
