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
            output_flat = output.view(-1, output.size(-1))
            trg_flat = trg[:, 1:].contiguous().view(-1)

            # Compute loss
            loss = loss_fn(output_flat, trg_flat)
            total_loss += loss.item()

            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else float('inf')


def evaluate_bleu(model: torch.nn.Module, val_loader: torch.utils.data.DataLoader,
                  trg_vocab: dict, device: torch.device) -> float:
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
    # print(f"Predictions (first 5): {predictions[:5]}")
    # print(f"References (first 5): {references[:5]}")

    # Handle empty references case
    if not references or not predictions or all(len(ref[0]) == 0 for ref in references):
        print("Error: No valid references or predictions for BLEU computation.")
        return 0.0  # Prevent division by zero

    # Compute BLEU score
    bleu_score = bleu_metric.compute(predictions=predictions, references=references)["bleu"]
    return bleu_score
