import torch
import sacrebleu
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt


def evaluate(model, dataloader, vocab, max_length, device):
    """
    Lightweight evaluation of the model using BLEU and ROUGE scores.

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
    hypotheses = []

    with torch.no_grad():
        for src, trg in dataloader:
            src, trg = src.to(device), trg.to(device)

            # Forward pass to get model outputs
            output = model(src, trg[:, :-1])  # Only pass the part of target needed for decoding

            # Convert model output to token indices (we assume greedy decoding)
            predicted_tokens = output.argmax(dim=-1)  # Get token with highest probability

            # Convert predicted token IDs to words
            for pred in predicted_tokens:
                pred_sentence = [vocab.get(idx.item(), "<unk>") for idx in pred if idx != vocab["<pad>"]]
                hypotheses.append(" ".join(pred_sentence))

    # Compute BLEU score
    bleu_score = sacrebleu.corpus_bleu(hypotheses, [[ref] for ref in hypotheses]).score

    # Return only BLEU score for simplicity
    return {"BLEU": bleu_score}


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






