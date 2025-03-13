import torch
import sacrebleu
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer

import matplotlib.pyplot as plt

from rouge_score import rouge_scorer
import sacrebleu

def evaluate(model, dataloader, vocab, max_length, device):
    """
    Evaluates the model using BLEU and ROUGE scores.

    Args:
        model (nn.Module): The trained Transformer model.
        dataloader (DataLoader): The validation or test data loader.
        vocab (dict): The target vocabulary mapping.
        max_length (int): Maximum sequence length.
        device (torch.device): The device (CPU/GPU).

    Returns:
        dict: BLEU and ROUGE scores.
    """
    model.eval()
    references = []
    hypotheses = []
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    with torch.no_grad():
        for src, trg in dataloader:
            src, trg = src.to(device), trg.to(device)

            # Generate translations (Greedy Decoding for now)
            predicted_tokens = model.greedy_decode(src, max_length)

            # Convert indices to words and collect references and hypotheses
            for ref, pred in zip(trg.tolist(), predicted_tokens.tolist()):
                ref_sentence = [word for idx in ref if (word := vocab.get(idx, "<unk>")) not in ["<pad>", "<bos>", "<eos>"]]
                pred_sentence = [word for idx in pred if (word := vocab.get(idx, "<unk>")) not in ["<pad>", "<bos>", "<eos>"]]

                references.append([" ".join(ref_sentence)])  # BLEU expects a list of reference sentences
                hypotheses.append(" ".join(pred_sentence))  # Single hypothesis

    # Compute BLEU score
    bleu_score = sacrebleu.corpus_bleu(hypotheses, [references]).score

    # Calculate ROUGE scores for each sentence
    rouge_scores = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for ref, hyp in zip(references, hypotheses):
        score = rouge.score(" ".join(ref), hyp)
        rouge_scores["rouge1"] += score["rouge1"].fmeasure
        rouge_scores["rouge2"] += score["rouge2"].fmeasure
        rouge_scores["rougeL"] += score["rougeL"].fmeasure

    # Average the ROUGE scores
    num_samples = len(hypotheses)
    rouge_scores = {key: value / num_samples for key, value in rouge_scores.items()}

    return {
        "BLEU": bleu_score,
        "ROUGE-1": rouge_scores["rouge1"],
        "ROUGE-2": rouge_scores["rouge2"],
        "ROUGE-L": rouge_scores["rougeL"],
    }


def plot_training_progress(history):
    epochs_range = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history["train_loss"], label="Train Loss", marker="o", color="blue")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss per Epoch")
    plt.legend()

    # Plot BLEU and ROUGE Scores
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history["bleu"], label="BLEU", marker="o", color="green", linestyle='--')
    plt.plot(epochs_range, history["rouge1"], label="ROUGE-1", marker="o", color="red", linestyle='-.')
    plt.plot(epochs_range, history["rougeL"], label="ROUGE-L", marker="o", color="purple", linestyle=':')

    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.title("BLEU and ROUGE Scores per Epoch")
    plt.legend()

    plt.tight_layout()
    plt.grid(True)
    plt.show()






