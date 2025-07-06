import torch
import evaluate as hf_evaluate


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
            logits = output.view(-1, output.size(-1))
            targets = trg[:, 1:].contiguous().view(-1)

            # Compute loss
            loss = loss_fn(logits, targets)
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

# def _translate_batch(model: torch.nn.Module, src: torch.Tensor) -> list[list[int]]:
#     """Translates a batch using the model and returns a list of token indices."""
#     return model.translate(src).cpu().tolist()
#
#
# def evaluate_bleu(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader,
#                   trg_vocab: dict[str, int], device: torch.device,
#                   verbose: bool = False,) -> float:
#     """Computes the BLEU score on a validation set using the model's translate method.
#
#     Args:
#         model (torch.nn.Module): The trained translation model.
#         data_loader (torch.utils.data.DataLoader): DataLoader containing the validation dataset.
#         trg_vocab (dict[str, int]): A dictionary mapping target language tokens to indices.
#         device (torch.device): The device to run the evaluation on (CPU or GPU).
#         verbose (bool, optional): If True, additional BLEU score details will be printed. Default is False.
#
#     Returns:
#         float: The computed BLEU score for the validation set.
#     """
#     bleu = hf_evaluate.load("bleu")
#     model.eval()
#     predictions, references = [], []
#
#     # Initialize token mapping and special tokens
#     idx_to_token = {idx: tok for tok, idx in trg_vocab.items()}
#     special_tokens = data_loader.dataset.special_tokens
#
#     with torch.no_grad():
#         for src, trg in data_loader:
#             src, trg = src.to(device), trg.to(device)
#             output = _translate_batch(model, src)
#
#             for pred_seq, ref_seq in zip(output, trg.cpu().tolist()):
#                 pred_tokens = _decode_sequence(pred_seq, idx_to_token)
#                 ref_tokens = _decode_sequence(ref_seq, idx_to_token)
#
#                 pred_str, ref_str_list = _format_for_bleu(pred_tokens, ref_tokens, special_tokens)
#
#                 if pred_str and ref_str_list[0]:  # Avoid empty strings
#                     predictions.append(pred_str)
#                     references.append(ref_str_list)
#
#     if not predictions or not references:
#         print("Warning: No valid predictions or references found. BLEU is 0.0.")
#         return 0.0
#
#     result = bleu.compute(predictions=predictions, references=references)
#
#     if verbose:
#         print(f"BLEU score: {result['bleu']:.4f}")
#         print(f"Precisions: {result.get('precisions')}")
#         print(f"Brevity Penalty: {result.get('brevity_penalty'):.4f}")
#
#     return result["bleu"]

def _translate_batch(model: torch.nn.Module, src: torch.Tensor,
                     beam_size: int = 2, max_len: int = 0) -> list[list[int]]:
    """Translates a batch with beam search and returns list of token indices."""
    return model.translate(src, beam_size=beam_size, max_len=max_len).cpu().tolist()

def evaluate_bleu(model: torch.nn.Module,
                  data_loader: torch.utils.data.DataLoader,
                  trg_vocab: dict[str, int],
                  device: torch.device,
                  special_tokens: list[str],
                  beam_size: int = 2,
                  max_len: int = 0,
                  verbose: bool = False) -> float:
    """
    Evaluates the model using BLEU score.

    Args:
        model (nn.Module): Translation model with a `translate` method.
        data_loader (DataLoader): Loader for validation/test data.
        trg_vocab (dict): Target vocabulary mapping tokens to indices.
        device (torch.device): Device (CPU or GPU).
        special_tokens (list[str]): Tokens like <pad>, <bos>, <eos> to ignore.
        beam_size (int): Beam size for beam search decoding.
        max_len (int): Max decoding length. 0 means auto-calculate.
        verbose (bool): Print BLEU components if True.

    Returns:
        float: BLEU score between 0 and 1.
    """
    bleu = hf_evaluate.load("bleu")
    model.eval()

    predictions, references = [], []
    idx_to_token = {idx: tok for tok, idx in trg_vocab.items()}
    special_token_set = set(special_tokens)

    with torch.no_grad():
        for src, trg in data_loader:
            src, trg = src.to(device), trg.to(device)

            output = _translate_batch(model, src, beam_size=beam_size, max_len=max_len)

            for pred_seq, ref_seq in zip(output, trg.cpu().tolist()):
                pred_tokens = _decode_sequence(pred_seq, idx_to_token)
                ref_tokens = _decode_sequence(ref_seq, idx_to_token)

                # Debug print!
                # print("Raw pred tokens:", pred_tokens)
                # print("Raw ref tokens:", ref_tokens)

                pred_str, ref_str_list = _format_for_bleu(pred_tokens, ref_tokens, special_token_set)

                # Debug print!
                # print("Cleaned pred:", pred_str)
                # print("Cleaned ref:", ref_str_list[0])

                if pred_str and ref_str_list[0]:
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
