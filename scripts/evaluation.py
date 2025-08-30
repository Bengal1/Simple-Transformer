"""
This module provides a suite of utility functions for evaluating the performance of
machine learning models, with a specific focus on sequence-to-sequence tasks
like machine translation.

The module includes functions for:
- Calculating the average loss over a dataset (`evaluate_model`).
- Post-processing model outputs, including decoding, removing special tokens,
  and detokenization (`_decode_sequence`, `_remove_special_tokens`, `_detokenize`).
- Computing the BLEU score, a standard metric for machine translation quality,
  using the `sacrebleu` library (`evaluate_bleu`).

These functions are designed to be used in conjunction with PyTorch models and
data loaders to streamline the evaluation process during training and testing.
"""
import torch
import sacrebleu
import logging
import re


# --- Public API ---
__all__ = [
    "evaluate_model",
    "evaluate_bleu"
]


# ------------------ Loss ------------------ #
def evaluate_model(model: torch.nn.Module,
                   data_loader: torch.utils.data.DataLoader,
                   loss_fn: torch.nn.modules.loss,
                   device: torch.device) -> float:
    """Evaluates the model on a given dataset using a loss function.

    Args:
        model (nn.Module): The transformer model to evaluate.
        data_loader (DataLoader): DataLoader for the validation or test dataset.
        loss_fn (Callable): The loss function used for evaluation.
        device (torch.device): The device (CPU or GPU) to run the evaluation on.

    Returns:
        float: The average loss over the dataset. Returns `float('inf')` if no batches
                are processed.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for src, trg in data_loader:
            src, trg = src.to(device), trg.to(device)
            # Forward pass
            output = model(src, trg[:, :-1])

            # Flatten the tensors for loss computation
            logits = output.view(-1, output.size(-1))
            targets = trg[:, 1:].contiguous().view(-1)

            # Compute loss
            loss = loss_fn(logits, targets)
            total_loss += loss.item()

            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else float('inf')


# ------------------ BLEU ------------------ #
def _decode_sequence(seq: list[int],
                     idx_to_token: dict[int, str]) -> list[str]:
    """
    Converts a sequence of token indices into a list of human-readable tokens.

    This method iterates through a list of integer token IDs and uses a provided
    mapping to convert each ID back into its corresponding string token. If an
    index is encountered that is not present in the `idx_to_token` dictionary,
    it defaults to the '<unk>' (unknown) token, ensuring robust decoding.

    Args:
        seq (list[int]): A list of integer indices representing a sequence of tokens.
        idx_to_token (dict[int, str]): A dictionary mapping integer indices to their
                                        corresponding string tokens.

    Returns:
        list[str]: A list of string tokens corresponding to the input
                    sequence of indices.
    """
    return [idx_to_token.get(idx, "<unk>") for idx in seq]


def _remove_special_tokens(tokens: list[str],
                           special_token_set: set[str]) -> list[str]:
    """
    Filters a list of tokens, removing any tokens that are considered 'special'.

    This utility function is useful for post-processing token sequences, such as
    removing padding tokens, start-of-sequence, or end-of-sequence markers,
    to obtain a clean sequence of content tokens.

    Args:
        tokens (list[str]): The input list of string tokens.
        special_token_set (set[str]): A set containing string representations of
                                       special tokens to be removed.

    Returns:
        list[str]: A new list containing only the tokens that were not present
                   in the `special_token_set`.
    """
    return [tok for tok in tokens if tok not in special_token_set]


def _detokenize(tokens: list[str]) -> str:
    """Reassembles a list of tokens into a single string, reversing
    the effects of tokenization.

    This function handles common punctuation and contractions that are often
    separated during tokenization. It ensures that apostrophes in contractions and
    various punctuation marks (.,?!:;) are correctly attached to the preceding word.
    It also cleans up spaces around quotation marks and parentheses.

    Args:
        tokens (list[str]): A list of string tokens to be joined.

    Returns:
        str: The detokenized string with correct spacing and punctuation.
    """
    text = " ".join(tokens)
    text = re.sub(r' (\'s|n\'t| \'m| \'ve| \'re| \'ll| \'d)', r'\1', text)
    text = re.sub(r' ([\.\,\?\!\:\;])', r'\1', text)
    text = re.sub(r' \( ', r' (', text)
    text = re.sub(r' \) ', r') ', text)
    text = re.sub(r' \" ', r' "', text)
    text = re.sub(r'\"', r'"', text)
    text = re.sub(r' ([\.\,\?\!\:\;])', r'\1', text)
    return text.strip()


# --- BLEU Evaluation Function ---

def evaluate_bleu(model: torch.nn.Module,
                  data_loader: torch.utils.data.DataLoader,
                  trg_vocab: dict[str, int],
                  device: torch.device,
                  special_tokens: list[str],
                  beam_size: int = 4,
                  max_len: int = 0,
                  verbose: bool = False) -> float:
    """
    Evaluates the model's translation performance using the BLEU metric via sacrebleu.

    Args:
        model (nn.Module): The Transformer model with a `translate` method.
                           `model.translate(src, beam_size, max_len)` should return
                           a tensor or list of lists of predicted token IDs.
        data_loader (torch.utils.data.DataLoader): DataLoader for the validation/test set.
        trg_vocab (Dict[str, int]): Target vocabulary mapping string tokens to integer IDs.
        device (torch.device): The device (CPU/GPU) to perform computations on.
        special_tokens (List[str]): A list of special token strings to be removed
                                from predictions and references before BLEU computation.
        beam_size (int): Beam width for the model's translation (decoding).
        max_len (int): Maximum length for generated sequences. If 0, `model.translate`
                       should handle this dynamically (e.g., based on source length).
        verbose (bool): If True, prints additional details about the BLEU result.

    Returns:
        float: The computed BLEU score, ranging from 0.0 to 100.0.
                Returns 0.0 if no valid predictions or references are found.
    """
    model.eval()  # Set model to evaluation mode

    all_predictions: list[str] = []
    all_references: list[str] = []

    # Create inverse mapping for decoding IDs to tokens
    idx_to_token = {idx: tok for tok, idx in trg_vocab.items()}
    # Convert special_tokens list to a set for efficient lookup
    special_token_set = set(special_tokens)

    with torch.no_grad():  # Disable gradient calculations for inference
        for src_batch, trg_batch in data_loader:
            src_batch = src_batch.to(device)
            trg_batch = trg_batch.to(device)

            predicted_ids_batch = model.translate(src_batch,
                                                  beam_size=beam_size,
                                                  max_len=max_len)

            # Convert tensors to standard Python lists for easier processing
            predicted_ids_batch = predicted_ids_batch.cpu().tolist()
            reference_ids_batch = trg_batch.cpu().tolist()

            # Process each sample in the current batch
            for predict_seq_ids, ref_seq_ids in zip(predicted_ids_batch,
                                                    reference_ids_batch):
                # Decode IDs to token strings
                decoded_pred = _decode_sequence(predict_seq_ids, idx_to_token)
                decoded_ref  = _decode_sequence(ref_seq_ids, idx_to_token)

                # Remove special tokens
                cleaned_pred_tokens = _remove_special_tokens(decoded_pred,
                                                             special_token_set)
                cleaned_ref_tokens  = _remove_special_tokens(decoded_ref,
                                                            special_token_set)

                if cleaned_pred_tokens and cleaned_ref_tokens:
                    all_predictions.append(_detokenize(cleaned_pred_tokens))
                    all_references.append(_detokenize(cleaned_ref_tokens))

    # Handle cases where no valid sequences were found
    if not all_predictions or not all_references:
        logging.warning("No valid predictions or references found for BLEU "
                        "calculation. Returning 0.0.")
        return 0.0

    bleu_result = sacrebleu.corpus_bleu(all_predictions,
                                        [all_references], tokenize='13a')

    if verbose:
        print("\n--- BLEU Score Details ---")
        print(f"BLEU score: {bleu_result.score:.3f}")
        print(
            f"Precisions (1-gram, 2-gram, 3-gram, 4-gram): {bleu_result.precisions}")
        print(f"Brevity Penalty: {bleu_result.bp:.4f}")
        print(f"Length Ratio: {bleu_result.ratio:.4f}")
        print(f"Translation Length: {bleu_result.sys_len}")
        print(f"Reference Length: {bleu_result.ref_len}")
        print("----------------------------")

    return bleu_result.score
