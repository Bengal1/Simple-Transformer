import json
from datasets import load_dataset

def make_iwslt14_local():
    # Load the dataset
    dataset = load_dataset("ahazeemi/iwslt14-en-fr")["train"]

    # Convert the dataset to a dictionary with "en" and "fr" as keys
    full_dataset = {"en": dataset["en"], "fr": dataset["fr"]}

    # Save the entire dataset to a JSON file
    with open("iwslt14_full.json", "w", encoding="utf-8") as f:
        json.dump(full_dataset, f, ensure_ascii=False, indent=4)

    print("Full dataset saved as iwslt14_full.json")


