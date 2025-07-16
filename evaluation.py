import torch
import evaluate as hf_evaluate


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
                     special_tokens: set[str]) -> tuple[str, list[str]]:
    """Cleans and joins token lists into BLEU-compatible string format."""
    cleaned_pred = _remove_special_tokens(decoded_pred, special_tokens)
    cleaned_ref = _remove_special_tokens(decoded_ref, special_tokens)
    return " ".join(cleaned_pred), [" ".join(cleaned_ref)]


def evaluate_bleu(model: torch.nn.Module,
                  data_loader: torch.utils.data.DataLoader,
                  trg_vocab: dict[str, int],
                  device: torch.device,
                  special_tokens: list[str],
                  beam_size: int = 2,
                  max_len: int = 0,
                  verbose: bool = False) -> float:
    """
    Evaluates the model on a dataset using BLEU score with beam search.

    Args:
        model (nn.Module): The Transformer model with `translate` method.
        data_loader (DataLoader): DataLoader for the test or validation split.
        trg_vocab (dict): Mapping from target tokens to indices.
        device (torch.device): Device to run evaluation on.
        special_tokens (list[str]): Special tokens to ignore (e.g., <pad>, <bos>, <eos>).
        beam_size (int): Number of beams for beam search.
        max_len (int): Max target length. If 0, it's computed dynamically.
        verbose (bool): Print BLEU components if True.

    Returns:
        float: The BLEU score between 0.0 and 1.0.
    """
    model.eval()
    bleu = hf_evaluate.load("bleu")
    predictions, references = [], []

    idx_to_token = {idx: tok for tok, idx in trg_vocab.items()}
    special_token_set = set(special_tokens)

    with torch.no_grad():
        for src_batch, trg_batch in data_loader:
            src_batch = src_batch.to(device)
            trg_batch = trg_batch.to(device)

            # === Generate predictions with beam search ===
            output_batch = model.translate(src_batch, beam_size=beam_size,
                                           max_len=max_len)
            output_batch = output_batch.cpu().tolist()
            trg_batch = trg_batch.cpu().tolist()

            for predict_seq, ref_seq in zip(output_batch, trg_batch):
                decoded_pred = _decode_sequence(predict_seq, idx_to_token)
                decoded_ref = _decode_sequence(ref_seq, idx_to_token)

                pred_str, ref_str_list = _format_for_bleu(decoded_pred, decoded_ref,
                                                          special_token_set)

                if pred_str and ref_str_list[0]:  # skip empty samples
                    predictions.append(pred_str)
                    references.append(ref_str_list)

    if not predictions or not references:
        print("Warning: No valid predictions or references found. BLEU is 0.0.")
        return 0.0

    result = bleu.compute(predictions=predictions, references=references)

    if verbose:
        print(f"\nBLEU score: {result['bleu']:.4f}")
        print(f"Precisions: {result.get('precisions')}")
        print(f"Brevity Penalty: {result.get('brevity_penalty'):.4f}")
        print(f"Length Ratio: {result.get('length_ratio'):.4f}")
        print(f"Translation Length: {result.get('translation_length')}")
        print(f"Reference Length: {result.get('reference_length')}")

    return result["bleu"]
