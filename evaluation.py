import torch
from nltk.translate.bleu_score import corpus_bleu
import evaluate as hf_evaluate
from tqdm.notebook import tqdm


def evaluate_model(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader,
                   loss_fn: torch.nn.modules.loss, device: torch.device) -> float:
    """Evaluates the model on a given dataset using a loss function.

    Args:
        model (nn.Module): The transformer model to evaluate.
        data_loader (DataLoader): DataLoader for the validation or test dataset.
        loss_fn (Callable): The loss function used for evaluation.
        device (torch.device): The device (CPU or GPU) to run the evaluation on.

    Returns:
        float: The average loss over the dataset. Returns `float('inf')` if no batches are processed.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for src, trg in data_loader:
            src, trg = src.to(device), trg.to(device)

            output = model(src, trg[:, :-1])  # Forward pass

            # Flatten the tensors for loss computation
            output_flat = output.view(-1, output.size(-1))
            trg_flat = trg[:, 1:].contiguous().view(-1)

            # Compute loss
            loss = loss_fn(output_flat, trg_flat)
            total_loss += loss.item()

            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else float('inf')



def _decode_sequence(seq: list[int], idx_to_token: dict[int, str]) -> list[str]:
    """Converts a list of token indices to a list of tokens."""
    return [idx_to_token.get(idx, "<unk>") for idx in seq]

def _remove_special_tokens(tokens: list[str], special_tokens: set[str]) -> list[str]:
    """Remove specified special tokens from a list of tokens."""
    return [tok for tok in tokens if tok not in special_tokens]

def _format_for_bleu(decoded_pred: list[str], decoded_ref: list[str],
                     special_tokens: set[str]) -> (str, list[str]):
    """Cleans and joins token lists into BLEU-compatible string format."""
    cleaned_pred = _remove_special_tokens(decoded_pred, special_tokens)
    cleaned_ref = _remove_special_tokens(decoded_ref, special_tokens)
    return " ".join(cleaned_pred), [" ".join(cleaned_ref)]

def _translate_batch(model: torch.nn.Module, src: torch.Tensor) -> list[list[int]]:
    """Translates a batch using the model and returns a list of token indices."""
    return model.translate(src).cpu().tolist()


def evaluate_bleu(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader,
                  trg_vocab: dict[str, int], device: torch.device,
                  verbose: bool = False,) -> float:
    """Computes the BLEU score on a validation set using the model's translate method.

    Args:
        model (torch.nn.Module): The trained translation model.
        data_loader (torch.utils.data.DataLoader): DataLoader containing the validation dataset.
        trg_vocab (dict[str, int]): A dictionary mapping target language tokens to indices.
        device (torch.device): The device to run the evaluation on (CPU or GPU).
        verbose (bool, optional): If True, additional BLEU score details will be printed. Default is False.

    Returns:
        float: The computed BLEU score for the validation set.
    """
    bleu = hf_evaluate.load("bleu")
    model.eval()
    predictions, references = [], []

    # Initialize token mapping and special tokens
    idx_to_token = {idx: tok for tok, idx in trg_vocab.items()}
    special_tokens = data_loader.dataset.special_tokens

    with torch.no_grad():
        for src, trg in tqdm(data_loader, desc="Evaluating BLEU", leave=False, dynamic_ncols=True):
            src, trg = src.to(device), trg.to(device)
            output = _translate_batch(model, src)

            for pred_seq, ref_seq in zip(output, trg.cpu().tolist()):
                pred_tokens = _decode_sequence(pred_seq, idx_to_token)
                ref_tokens = _decode_sequence(ref_seq, idx_to_token)

                pred_str, ref_str_list = _format_for_bleu(pred_tokens, ref_tokens, special_tokens)

                if pred_str and ref_str_list[0]:  # Avoid empty strings
                    predictions.append(pred_str)
                    references.append(ref_str_list)

    if not predictions or not references:
        print("Warning: No valid predictions or references found. BLEU is 0.0.")
        return 0.0

    result = bleu.compute(predictions=predictions, references=references)

    if verbose:
        print(f"BLEU score: {result['bleu']:.4f}")
        print(f"Precisions: {result.get('precisions')}")
        print(f"Brevity Penalty: {result.get('brevity_penalty'):.4f}")

    return result["bleu"]


# def evaluate_bleu(model: torch.nn.Module, val_loader: torch.utils.data.DataLoader,
#                   trg_vocab: dict, device: torch.device) -> float:
#     """Computes the BLEU score for the model using Hugging Face's evaluate library.
#
#     Args:
#         model (nn.Module): The transformer model used for translation.
#         val_loader (DataLoader): DataLoader for the validation dataset.
#         trg_vocab (dict): Target language vocabulary mapping from tokens to indices.
#         device (torch.device): The device (CPU or GPU) to run the evaluation on.
#
#     Returns:
#         float: The BLEU score computed over the validation set.
#     """
#     bleu_metric = hf_evaluate.load("bleu")
#
#     model.eval()
#     predictions = []
#     references = []
#
#     idx_to_token = {idx: token for token, idx in trg_vocab.items()}  # Reverse vocab mapping
#
#     with torch.no_grad():
#         for src, trg in val_loader:
#             src, trg = src.to(device), trg.to(device)
#             output = model.translate(src)
#             output = output.cpu().tolist()
#             print("Model output:", output[:3]) # Debug
#             # Convert token indices to words
#             decoded_sentences = [[idx_to_token.get(idx, "<unk>") for idx in sent] for sent in output]
#             reference_sentences = [[idx_to_token.get(idx, "<unk>") for idx in sent.tolist()] for sent in trg]
#             # print(f"Output: {decoded_sentences}") # Debug
#
#             # Remove special tokens
#             decoded_sentences = [[word for word in sent if word not in ["<bos>", "<eos>", "<pad>"]] for sent in
#                                  decoded_sentences]
#             reference_sentences = [[word for word in sent if word not in ["<bos>", "<eos>", "<pad>"]] for sent in
#                                    reference_sentences]
#             # Debug
#             # print("Decoded:", decoded_sentences[:3])
#             # print("Refs:", reference_sentences[:3])
#
#             predictions.extend([" ".join(sent) for sent in decoded_sentences])
#             references.extend([[" ".join(sent)] for sent in reference_sentences])
#
#     # Debug
#     # print(f"Predictions (first 5): {predictions[:5]}")
#     # print(f"References (first 5): {references[:5]}")
#
#     # Handle empty references case
#     if not references or not predictions or all(len(ref[0]) == 0 for ref in references):
#         print("Error: No valid references or predictions for BLEU computation.")
#         return 0.0  # Prevent division by zero
#
#     # Compute BLEU score
#     bleu_score = bleu_metric.compute(predictions=predictions, references=references)["bleu"]
#     return bleu_score
