import torch
import evaluate as hf_evaluate

def evaluate_model(model, data_loader, loss_fn, device):
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

            # Shift target: replace <eos> (ID=3) with <pad> (ID=1)
            shifted_trg = trg.clone()
            shifted_trg[shifted_trg == 3] = 1

            output = model(src, shifted_trg)  # Forward pass
            target = trg.reshape(-1)

            loss = loss_fn(output.view(-1, output.shape[-1]), target)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else float('inf')


def evaluate_bleu(model, val_loader, trg_vocab, device):
    """Computes the BLEU score for the model using Hugging Face's evaluate library.

    Args:
        model (nn.Module): The transformer model used for translation.
        val_loader (DataLoader): DataLoader for the validation dataset.
        trg_vocab (dict): Target language vocabulary mapping from tokens to indices.
        device (torch.device): The device (CPU or GPU) to run the evaluation on.

    Returns:
        float: The BLEU score computed over the validation set.
    """
    bleu_metric = hf_evaluate.load("bleu")

    model.eval()
    predictions = []
    references = []

    idx_to_token = {idx: token for token, idx in trg_vocab.items()}  # Reverse vocab mapping

    with torch.no_grad():
        for src, trg in val_loader:
            src, trg = src.to(device), trg.to(device)
            output = model.translate(src)
            output = output.cpu().tolist()

            # Convert token indices to words
            decoded_sentences = [[idx_to_token.get(idx, "<unk>") for idx in sent] for sent in output]
            reference_sentences = [[idx_to_token.get(idx, "<unk>") for idx in sent.tolist()] for sent in trg]
            # print(f"Output: {decoded_sentences}") # Debug

            # Remove special tokens
            decoded_sentences = [[word for word in sent if word not in ["<bos>", "<eos>", "<pad>"]] for sent in
                                 decoded_sentences]
            reference_sentences = [[word for word in sent if word not in ["<bos>", "<eos>", "<pad>"]] for sent in
                                   reference_sentences]

            predictions.extend([" ".join(sent) for sent in decoded_sentences])
            references.extend([[" ".join(sent)] for sent in reference_sentences])

    # Debug
    print(f"Predictions (first 5): {predictions[:5]}")
    print(f"References (first 5): {references[:5]}")

    # Handle empty references case
    if not references or not predictions or all(len(ref[0]) == 0 for ref in references):
        print("Error: No valid references or predictions for BLEU computation.")
        return 0.0  # Prevent division by zero

    # Compute BLEU score
    bleu_score = bleu_metric.compute(predictions=predictions, references=references)["bleu"]
    return bleu_score


# ------------------------------------------------------------------------------ #

# def evaluate_bleu(model, data_loader, vocab, device):
#     predictions = []
#     references = []
#
#     for src, trg in data_loader:
#         src = src.to(device)
#         trg = trg.to(device)
#
#         # Translate the source sequence to target language
#         translated = model.translate(src)
#
#         # Convert translated tokens back to words (assuming you have a function to do this)
#         translated_words = vocab.decode(translated.squeeze(0).cpu().numpy())
#
#         # Check if translation is empty, and handle it
#         if not translated_words:
#             translated_words = ["<unk>"] * len(trg[0])  # Fallback to <unk> if empty
#
#         predictions.append(translated_words)
#         references.append([trg])
#
#     # Compute BLEU score
#     bleu_score = bleu_metric.compute(predictions=predictions, references=references)["bleu"]
#     return bleu_score
#
#
# from collections import Counter
#
# def compute_bleu_safe(predictions, references):
#     # Ensure no empty predictions or references
#     valid_predictions = [p for p in predictions if len(p) > 0]
#     valid_references = [r for r in references if len(r) > 0]
#
#     if not valid_predictions or not valid_references:
#         return 0.0  # Return 0 BLEU score if there are no valid predictions
#
#     return bleu_metric.compute(predictions=valid_predictions, references=valid_references)["bleu"]
