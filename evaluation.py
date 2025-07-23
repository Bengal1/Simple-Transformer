import torch
# import evaluate as hf_evaluate
import sacrebleu
import logging

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
    """
    Converts a list of token indices to a list of tokens (strings).
    Handles potential out-of-vocabulary indices by mapping to '<unk>'.
    """
    return [idx_to_token.get(idx, "<unk>") for idx in seq]


def _remove_special_tokens(tokens: list[str], special_token_set: set[str]) -> list[
    str]:
    """
    Removes specified special tokens from a list of tokens.
    """
    return [tok for tok in tokens if tok not in special_token_set]


# --- BLEU Evaluation Function ---

def evaluate_bleu(model: torch.nn.Module,
                  data_loader: torch.utils.data.DataLoader,
                  trg_vocab: dict[str, int],
                  device: torch.device,
                  special_tokens: list[str],
                  beam_size: int = 2,
                  max_len: int = 0,
                  # If 0, model.translate should determine dynamically
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

    all_predictions_joined: list[str] = []
    all_references_joined: list[list[str]] = []

    # Create inverse mapping for decoding IDs to tokens
    idx_to_token = {idx: tok for tok, idx in trg_vocab.items()}
    # Convert special_tokens list to a set for efficient lookup
    special_token_set = set(special_tokens)

    with torch.no_grad():  # Disable gradient calculations for inference
        for src_batch, trg_batch in data_loader:
            src_batch = src_batch.to(device)
            trg_batch = trg_batch.to(device)

            predicted_ids_batch = model.translate(src_batch, beam_size=beam_size,
                                                  max_len=max_len)

            # Convert tensors to standard Python lists for easier processing
            predicted_ids_batch = predicted_ids_batch.cpu().tolist()
            reference_ids_batch = trg_batch.cpu().tolist()

            # Process each sample in the current batch
            for predict_seq_ids, ref_seq_ids in zip(predicted_ids_batch,
                                                    reference_ids_batch):
                # 1. Decode IDs to token strings
                decoded_pred = _decode_sequence(predict_seq_ids, idx_to_token)
                decoded_ref = _decode_sequence(ref_seq_ids, idx_to_token)

                # 2. Remove special tokens
                cleaned_pred_tokens = _remove_special_tokens(decoded_pred,
                                                             special_token_set)
                cleaned_ref_tokens = _remove_special_tokens(decoded_ref,
                                                            special_token_set)

                if cleaned_pred_tokens and cleaned_ref_tokens:
                    all_predictions_joined.append(" ".join(cleaned_pred_tokens))
                    all_references_joined.append([" ".join(cleaned_ref_tokens)])

    # Handle cases where no valid sequences were generated/found
    if not all_predictions_joined or not all_references_joined:
        print(
            "Warning: No valid predictions or references found for BLEU calculation. Returning 0.0.")
        return 0.0

    bleu_result = sacrebleu.corpus_bleu(all_predictions_joined,
                                        all_references_joined, tokenize='none')

    if verbose:
        print("\n--- BLEU Score Details ---")
        print(f"BLEU score: {bleu_result.score:.2f}")
        print(
            f"Precisions (1-gram, 2-gram, 3-gram, 4-gram): {bleu_result.precisions}")
        print(f"Brevity Penalty: {bleu_result.bp:.4f}")
        print(f"Length Ratio: {bleu_result.ratio:.4f}")
        print(f"Translation Length: {bleu_result.sys_len}")
        print(f"Reference Length: {bleu_result.ref_len}")
        print("--------------------------")

    return bleu_result.score


# def _decode_sequence(seq: list[int], idx_to_token: dict[int, str]) -> list[str]:
#     """Converts a list of token indices to a list of tokens."""
#     return [idx_to_token.get(idx, "<unk>") for idx in seq]
#
#
# def _remove_special_tokens(tokens: list[str], special_tokens: set[str]) -> list[str]:
#     """Remove specified special tokens from a list of tokens."""
#     return [tok for tok in tokens if tok not in special_tokens]
#
# def _format_for_bleu(decoded_pred: list[str], decoded_ref: list[str],
#                      special_tokens: set[str]) -> tuple[list[str], list[list[str]]]: # Changed return type
#     """Cleans token lists for BLEU computation (returns lists of tokens)."""
#     cleaned_pred = _remove_special_tokens(decoded_pred, special_tokens)
#     cleaned_ref = _remove_special_tokens(decoded_ref, special_tokens)
#     # Return lists of tokens, not joined strings
#     return cleaned_pred, [cleaned_ref] # References are a list of lists of tokens
#
#
# def evaluate_bleu(model: torch.nn.Module,
#                      data_loader: torch.utils.data.DataLoader,
#                      trg_vocab: dict[str, int],
#                      device: torch.device,
#                      special_tokens: list[str],
#                      beam_size: int = 2,
#                      max_len: int = 0,
#                      verbose: bool = False) -> float:
#     model.eval()
#     bleu = hf_evaluate.load("bleu")
#     predictions, references = [], []
#
#     idx_to_token = {idx: tok for tok, idx in trg_vocab.items()}
#     special_token_set = set(special_tokens)
#
#     with torch.no_grad():
#         for src_batch, trg_batch in data_loader:
#             src_batch = src_batch.to(device)
#             trg_batch = trg_batch.to(device)
#
#             output_batch = model.translate(src_batch, beam_size=beam_size, max_len=max_len)
#             output_batch = output_batch.cpu().tolist()
#             trg_batch = trg_batch.cpu().tolist()
#
#             # # --- ADD THESE PRINT STATEMENTS ---
#             # print("\n--- DEBUGGING BLEU ---")
#             # print(f"Example raw prediction IDs: {output_batch[0]}")
#             # print(f"Example raw reference IDs: {trg_batch[0]}")
#             #
#             # decoded_pred_sample = _decode_sequence(output_batch[0], idx_to_token)
#             # decoded_ref_sample = _decode_sequence(trg_batch[0], idx_to_token)
#             # print(f"Example decoded prediction: {decoded_pred_sample}")
#             # print(f"Example decoded reference: {decoded_ref_sample}")
#             #
#             # cleaned_pred_sample = _remove_special_tokens(decoded_pred_sample,
#             #                                              special_token_set)
#             # cleaned_ref_sample = _remove_special_tokens(decoded_ref_sample,
#             #                                             special_token_set)
#             # print(f"Example cleaned prediction: {cleaned_pred_sample}")
#             # print(f"Example cleaned reference: {cleaned_ref_sample}")
#             # # --- END DEBUGGING PRINTS ---
#
#             for predict_seq, ref_seq in zip(output_batch, trg_batch):
#                 decoded_pred = _decode_sequence(predict_seq, idx_to_token)
#                 decoded_ref = _decode_sequence(ref_seq, idx_to_token)
#
#                 # Now passing lists of tokens directly
#                 pred_tokens, ref_tokens_list = _format_for_bleu(decoded_pred, decoded_ref, special_token_set)
#
#                 # Check for empty lists after cleaning
#                 if pred_tokens and ref_tokens_list[0]: # Ensure both pred and ref are not empty
#                     predictions.append(pred_tokens)
#                     references.append(ref_tokens_list)
#
#     if not predictions or not references:
#         print("Warning: No valid predictions or references found. BLEU is 0.0.")
#         return 0.0
#
#     result = bleu.compute(predictions=predictions, references=references) # <-- Now inputs are list[list[str]]
#
#     if verbose:
#         print(f"\nBLEU score: {result['bleu']:.4f}")
#         print(f"Precisions: {result.get('precisions')}")
#         print(f"Brevity Penalty: {result.get('brevity_penalty'):.4f}")
#         print(f"Length Ratio: {result.get('length_ratio'):.4f}")
#         print(f"Translation Length: {result.get('translation_length')}")
#         print(f"Reference Length: {result.get('reference_length')}")
#
#     return result["bleu"]
#
# # def _format_for_bleu(decoded_pred: list[str], decoded_ref: list[str],
# #                      special_tokens: set[str]) -> tuple[str, list[str]]:
# #     """Cleans and joins token lists into BLEU-compatible string format."""
# #     cleaned_pred = _remove_special_tokens(decoded_pred, special_tokens)
# #     cleaned_ref = _remove_special_tokens(decoded_ref, special_tokens)
# #     return " ".join(cleaned_pred), [" ".join(cleaned_ref)]
#
#
# # def evaluate_bleu(model: torch.nn.Module,
# #                   data_loader: torch.utils.data.DataLoader,
# #                   trg_vocab: dict[str, int],
# #                   device: torch.device,
# #                   special_tokens: list[str],
# #                   beam_size: int = 2,
# #                   max_len: int = 0,
# #                   verbose: bool = False) -> float:
# #     """
# #     Evaluates the model on a dataset using BLEU score with beam search.
# #
# #     Args:
# #         model (nn.Module): The Transformer model with `translate` method.
# #         data_loader (DataLoader): DataLoader for the test or validation split.
# #         trg_vocab (dict): Mapping from target tokens to indices.
# #         device (torch.device): Device to run evaluation on.
# #         special_tokens (list[str]): Special tokens to ignore (e.g., <pad>, <bos>, <eos>).
# #         beam_size (int): Number of beams for beam search.
# #         max_len (int): Max target length. If 0, it's computed dynamically.
# #         verbose (bool): Print BLEU components if True.
# #
# #     Returns:
# #         float: The BLEU score between 0.0 and 1.0.
# #     """
# #     model.eval()
# #     bleu = hf_evaluate.load("bleu")
# #     predictions, references = [], []
# #
# #     idx_to_token = {idx: tok for tok, idx in trg_vocab.items()}
# #     special_token_set = set(special_tokens)
# #
# #     with torch.no_grad():
# #         for src_batch, trg_batch in data_loader:
# #             src_batch = src_batch.to(device)
# #             trg_batch = trg_batch.to(device)
# #
# #             # === Generate predictions with beam search ===
# #             output_batch = model.translate(src_batch, beam_size=beam_size,
# #                                            max_len=max_len)
# #             output_batch = output_batch.cpu().tolist()
# #             trg_batch = trg_batch.cpu().tolist()
# #
# #             for predict_seq, ref_seq in zip(output_batch, trg_batch):
# #                 decoded_pred = _decode_sequence(predict_seq, idx_to_token)
# #                 decoded_ref = _decode_sequence(ref_seq, idx_to_token)
# #
# #                 pred_str, ref_str_list = _format_for_bleu(decoded_pred, decoded_ref,
# #                                                           special_token_set)
# #
# #                 if pred_str and ref_str_list[0]:  # skip empty samples
# #                     predictions.append(pred_str)
# #                     references.append(ref_str_list)
# #
# #     if not predictions or not references:
# #         print("Warning: No valid predictions or references found. BLEU is 0.0.")
# #         return 0.0
# #
# #     result = bleu.compute(predictions=predictions, references=references)
# #
# #     if verbose:
# #         print(f"\nBLEU score: {result['bleu']:.4f}")
# #         print(f"Precisions: {result.get('precisions')}")
# #         print(f"Brevity Penalty: {result.get('brevity_penalty'):.4f}")
# #         print(f"Length Ratio: {result.get('length_ratio'):.4f}")
# #         print(f"Translation Length: {result.get('translation_length')}")
# #         print(f"Reference Length: {result.get('reference_length')}")
# #
# #     return result["bleu"]
