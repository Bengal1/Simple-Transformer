import torch
import evaluate as hf_evaluate


def evaluate_model(model, val_loader, trg_vocab, device):
    """Evaluates the model and computes the BLEU score using Hugging Face evaluate."""
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
            print(f"Output: {decoded_sentences}") # Debug

            # Remove special tokens
            decoded_sentences = [[word for word in sent if word not in ["<bos>", "<eos>", "<pad>"]] for sent in
                                 decoded_sentences]
            reference_sentences = [[word for word in sent if word not in ["<bos>", "<eos>", "<pad>"]] for sent in
                                   reference_sentences]

            predictions.extend([" ".join(sent) for sent in decoded_sentences])
            references.extend([[" ".join(sent)] for sent in reference_sentences])

    # print(f"Predictions: {predictions}") # Debug
    # print(f"References: {references}") # Debug

    # Compute BLEU score
    bleu_score = bleu_metric.compute(predictions=predictions, references=references)["bleu"]
    return bleu_score
