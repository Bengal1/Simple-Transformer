import torch
import evaluate as hf_evaluate


def evaluate_model(model, val_loader, trg_vocab, max_length, device):
    """Evaluates the model and computes the BLEU score using Hugging Face evaluate."""
    bleu_metric = hf_evaluate.load("bleu")

    model.eval()
    predictions = []
    references = []

    idx_to_token = {idx: token for token, idx in trg_vocab.items()}  # Reverse vocab mapping

    with torch.no_grad():
        for src, trg in val_loader:
            src, trg = src.to(device), trg.to(device)
            batch_size = src.shape[0]
            decoded_sentences = [["<bos>"] for _ in range(batch_size)]

            for _ in range(max_length):
                trg_tensor = torch.tensor(
                    [[trg_vocab.get(token, 0) for token in sent] for sent in decoded_sentences],
                    dtype=torch.long, device=device
                )
                output = model.translate(src)
                next_tokens = output.argmax(-1)[:, -1].tolist()

                for i, token in enumerate(next_tokens):
                    decoded_sentences[i].append(token)
                    if token == trg_vocab["<eos>"]:
                        decoded_sentences[i] = decoded_sentences[i][1:]  # Remove <bos>

            # Convert token indices back to words
            decoded_sentences = [[idx_to_token.get(idx, "<unk>") for idx in sent] for sent in decoded_sentences]
            reference_sentences = [[idx_to_token.get(idx, "<unk>") for idx in sent.tolist()] for sent in trg.cpu()]

            # Remove special tokens (<bos>, <eos>, <pad>)
            decoded_sentences = [[word for word in sent if word not in ["<bos>", "<eos>", "<pad>"]] for sent in decoded_sentences]
            reference_sentences = [[word for word in sent if word not in ["<bos>", "<eos>", "<pad>"]] for sent in reference_sentences]

            predictions.extend([" ".join(sent) for sent in decoded_sentences])
            references.extend([[" ".join(sent)] for sent in reference_sentences])

    # Compute BLEU score
    bleu_score = bleu_metric.compute(predictions=predictions, references=references)["bleu"]
    return bleu_score

